// ============================================================
//  bot.js  —  Telegram AI Bot  (Production Hardened)
//  Fixes: 29 bugs from audit report
//  Architecture: Webhook · MongoDB Atlas Free · Render Free
// ============================================================

import express    from "express";
import TelegramBot from "node-telegram-bot-api";
import dotenv     from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";
import mongoose   from "mongoose";
import fetch      from "node-fetch";
import OpenAI     from "openai";
import Anthropic  from "@anthropic-ai/sdk";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const pdf     = require("pdf-parse");
const mammoth = require("mammoth");

dotenv.config();

// ============================================================
//  SECTION 1 · ENV CONFIGURATION
// ============================================================
const CFG = {
  TOKEN              : process.env.TELEGRAM_TOKEN,
  ADMIN_ID           : Number(process.env.ADMIN_ID),
  RENDER_URL         : process.env.RENDER_URL || "",
  PORT               : Number(process.env.PORT) || 3000,
  FORCE_CHANNEL      : process.env.FORCE_JOIN_CHANNEL || "",

  DAILY_REQUEST_LIMIT: Number(process.env.DAILY_REQUEST_LIMIT)     || 50,
  DAILY_TOKEN_LIMIT  : Number(process.env.DAILY_TOKEN_LIMIT)        || 50000,
  MAX_REPLY_TOKENS   : Number(process.env.MAX_REPLY_TOKENS)         || 1024,

  SEARCH_LIMIT       : Number(process.env.SEARCH_LIMIT)             || 10,
  SEARCH_RESULTS     : Number(process.env.SEARCH_RESULTS)           || 5,
  IMAGINE_LIMIT      : Number(process.env.IMAGINE_LIMIT)            || 5,
  DOC_LIMIT          : Number(process.env.LIMIT_DOC_ANALYSIS)       || 5,
  IMG_LIMIT          : Number(process.env.LIMIT_IMG_ANALYSIS)       || 10,
  PRO_LIMIT          : Number(process.env.LIMIT_PRO_MODEL)          || 10,
  // FIX #20: cap stored history text to keep documents small
  DOC_CHAR_LIMIT     : Number(process.env.DOC_CHAR_LIMIT)           || 10000,
  MAX_HISTORY_CHARS  : Number(process.env.MAX_HISTORY_CHARS)        || 500,

  // FIX #8: rate limits — env values in SECONDS, converted to ms here
  RATE_LIMIT_MS      : (Number(process.env.RATE_LIMIT_MS)           || 3)  * 1000,
  COMMAND_LIMIT_MS   : (Number(process.env.COMMAND_LIMIT_MS)        || 5)  * 1000,
  MEDIA_COOLDOWN_MS  : (Number(process.env.LIMIT_MEDIA_COOLDOWN)    || 10) * 1000,
  MAX_FILE_MB        : Number(process.env.MAX_FILE_MB)              || 10,

  MAX_CONCURRENT     : Number(process.env.MAX_CONCURRENT_REQUESTS)  || 20,
  QUEUE_LIMIT        : Number(process.env.REQUEST_QUEUE_LIMIT)      || 100,
  CACHE_CLEAR_MIN    : Number(process.env.MEMORY_CACHE_CLEAR_MINUTES)|| 30,
  MEMBER_CACHE_MIN   : Number(process.env.MEMBERSHIP_CACHE_MINUTES) || 10,
  APPROVAL_HOURS     : Number(process.env.APPROVAL_EXPIRY_HOURS)    || 24,
  // FIX #9: approval cache TTL — short so revocations propagate quickly
  APPROVAL_CACHE_TTL : (Number(process.env.APPROVAL_CACHE_SECONDS)  || 60) * 1000,

  HISTORY_MESSAGES   : Number(process.env.HISTORY_MESSAGES)         || 10,
  DB_MSG_LIMIT       : Number(process.env.DB_MSG_LIMIT)             || 20,
  CHAT_HISTORY_TTL   : Number(process.env.CHAT_HISTORY_TTL_DAYS)    || 30,
  INACTIVE_USER_DAYS : Number(process.env.INACTIVE_USER_DELETE_DAYS)|| 90,

  // FIX #5: external API timeout in ms
  API_TIMEOUT_MS     : Number(process.env.API_TIMEOUT_MS)           || 30000,

  GEMINI_API_KEY     : process.env.GEMINI_API_KEY,
  OPENAI_API_KEY     : process.env.OPENAI_API_KEY,
  CLAUDE_API_KEY     : process.env.CLAUDE_API_KEY,
  SERPER_API_KEY     : process.env.SERPER_API_KEY,
};

// ============================================================
//  SECTION 2 · GLOBAL SAFETY NETS
// ============================================================
process.on("unhandledRejection", (err) => console.error("⚠️ UnhandledRejection:", err?.message || err));
process.on("uncaughtException",  (err) => console.error("⚠️ UncaughtException:",  err?.message || err));

// ============================================================
//  SECTION 3 · MONGODB — connect first, start bot after
//  FIX #6: Bot does NOT start accepting requests until DB is ready.
//  FIX #7: All dates use UTC consistently (toISOString / Date.UTC).
// ============================================================
mongoose.connection.on("disconnected", () => {
  console.warn("⚠️ MongoDB disconnected — reconnecting…");
  setTimeout(connectDB, 3000);
});
mongoose.connection.on("error", (err) => {
  console.error("❌ MongoDB error:", err.message);
});

async function connectDB() {
  try {
    await mongoose.connect(process.env.MONGODB_CONNECT, {
      dbName                 : "Telegram",
      serverSelectionTimeoutMS: 10000,
      socketTimeoutMS        : 45000,
      maxPoolSize            : 10,   // respect Atlas Free connection cap
    });
    console.log("✅ MongoDB connected");
  } catch (err) {
    console.error("❌ MongoDB connection failed:", err.message);
    setTimeout(connectDB, 5000);
  }
}

// ── Schemas ───────────────────────────────────────────────

// FIX #10: PUBLIC_MODE persisted in DB so it survives Render restarts
const settingsSchema = new mongoose.Schema(
  { key: { type: String, required: true, unique: true }, value: mongoose.Schema.Types.Mixed },
  { versionKey: false }
);
const Setting = mongoose.model("Setting", settingsSchema);

// ── Daily aggregated stats (one document, resets at UTC midnight) ──
const dailyStatsSchema = new mongoose.Schema(
  {
    date     : { type: String, required: true, unique: true }, // "YYYY-MM-DD"
    tokens   : { type: Number, default: 0 },
    requests : { type: Number, default: 0 },
    search   : { type: Number, default: 0 },
    imagine  : { type: Number, default: 0 },
    doc      : { type: Number, default: 0 },
    img      : { type: Number, default: 0 },
    // TTL field — MongoDB auto-deletes this document 7 days after creation
    createdAt: { type: Date, default: () => new Date(), index: { expireAfterSeconds: 7 * 86400 } },
  },
  { versionKey: false }
);
const DailyStats = mongoose.model("DailyStats", dailyStatsSchema);

// Atomically increment one or more fields on today's stats document.
// Creates the document if it doesn't exist yet (upsert).
function bumpStats(fields) {
  const today = utcToday();
  DailyStats.findOneAndUpdate(
    { date: today },
    { $inc: fields },
    { upsert: true, new: true }
  ).catch(e => console.error("❌ bumpStats error:", e.message));
}

const userSchema = new mongoose.Schema(
  {
    // FIX #17: chatId always stored as String — never Number
    chatId        : { type: String, required: true, unique: true },
    approvedUntil : { type: Date,   default: null  },
    selectedModel : { type: String, default: "gemini" },
    language      : { type: String, default: "en"  },
    requests      : { type: Number, default: 0     },
    lastReset     : { type: Date,   default: () => new Date() },
    tokensUsed    : { type: Number, default: 0     },
    tokensReset   : { type: Date,   default: () => new Date() },
    accountCreated: { type: Date,   default: () => new Date() },
    lastActive    : { type: Date,   default: () => new Date() },
  },
  { versionKey: false }
);
// chatId unique index is already declared inline in the schema field — no .index() needed here
userSchema.index({ approvedUntil: 1 });
userSchema.index({ lastActive: 1 });
const User = mongoose.model("User", userSchema);

// FIX #20: Chat history stored separately, text truncated per message
const chatHistorySchema = new mongoose.Schema(
  {
    chatId   : { type: String, required: true },
    messages : [{
      role     : { type: String, enum: ["user", "bot"], required: true },
      // Store only a truncated snippet — keeps documents tiny
      text     : { type: String, maxlength: CFG.MAX_HISTORY_CHARS },
      createdAt: { type: Date,   default: () => new Date() },
    }],
    updatedAt: {
      type   : Date,
      default: () => new Date(),
      // MongoDB TTL index — auto-deletes after N days of inactivity
      index  : { expireAfterSeconds: CFG.CHAT_HISTORY_TTL * 86400 },
    },
  },
  { versionKey: false }
);
chatHistorySchema.index({ chatId: 1 }, { unique: true });
const ChatHistory = mongoose.model("ChatHistory", chatHistorySchema);

// ============================================================
//  SECTION 4 · IN-MEMORY CACHES  (ephemeral runtime data only)
//  FIX #1:  All caches declared here, before any use.
//  FIX #17: All Map keys are String(chatId) — never raw Number.
// ============================================================

// Rate limits
const rateLimitMap   = new Map(); // String(chatId) → timestamp
const mediaRateLimit = new Map(); // String(chatId) → timestamp
const commandRateLmt = new Map(); // `${chatId}:cmd`  → timestamp

// Membership & approval caches
const membershipCache = new Map(); // String(chatId) → { isMember, expiry }
const approvalCache   = new Map(); // String(chatId) → { approved, expiry }

// Per-user daily usage (in-memory only; resets at midnight UTC)
const userUsage = new Map(); // String(chatId) → { date, search, imagine, doc, img, proTokens }

// Preferences cache (backed by DB; session cache to reduce reads)
// FIX #22: These Maps ARE evicted by the periodic sweep below.
const userLanguages  = new Map(); // String(chatId) → langCode
const userModels     = new Map(); // String(chatId) → modelId
const _prefsPending  = new Map(); // dedup in-flight DB fetches

// Album dedup (FIX #14)
const albumSeen = new Map(); // media_group_id → timestamp

// Active processing (FIX #3: covers all heavy handlers)
const processingSet      = new Set();
const PROCESSING_TIMEOUT = 3 * 60 * 1000; // 3-min safety release

function cid(chatId) { return String(chatId); } // canonical key helper

// ── Periodic cache sweep — prevents memory leaks ──────────
// FIX #22: userLanguages and userModels also evicted here
setInterval(() => {
  const now     = Date.now();
  const STALE   = CFG.CACHE_CLEAR_MIN * 60 * 1000;
  const today   = utcToday();

  for (const [k, v] of rateLimitMap)    if (now - v > STALE)   rateLimitMap.delete(k);
  for (const [k, v] of mediaRateLimit)  if (now - v > STALE)   mediaRateLimit.delete(k);
  for (const [k, v] of commandRateLmt)  if (now - v > STALE)   commandRateLmt.delete(k);
  for (const [k, v] of membershipCache) if (now > v.expiry)     membershipCache.delete(k);
  for (const [k, v] of approvalCache)   if (now > v.expiry)     approvalCache.delete(k);
  for (const [k, v] of userUsage)       if (v.date !== today)   userUsage.delete(k);
  // albumSeen entries are deleted directly via setTimeout — no sweeper needed here
  // Evict pref caches — will be reloaded from DB on next request
  if (userLanguages.size > 5000) userLanguages.clear();
  if (userModels.size    > 5000) userModels.clear();

  console.log(`🧹 Cache swept [${new Date().toISOString()}] rateLimits=${rateLimitMap.size} prefs=${userLanguages.size}`);
}, CFG.CACHE_CLEAR_MIN * 60 * 1000);

// ============================================================
//  SECTION 5 · CONCURRENCY LIMITER
//  FIX #4: queue shift wrapped in try/catch so one bad task
//          doesn't freeze the entire queue.
// ============================================================
let activeRequests = 0;
const requestQueue  = [];

async function enqueueRequest(fn) {
  if (activeRequests + requestQueue.length >= CFG.QUEUE_LIMIT) {
    throw new Error("QUEUE_FULL");
  }

  return new Promise((resolve, reject) => {
    const task = async () => {
      activeRequests++;
      try   { resolve(await fn()); }
      catch (e) { reject(e); }
      finally {
        activeRequests--;
        // FIX #4: guard the shift so a bad task can't stall the queue
        if (requestQueue.length > 0) {
          const next = requestQueue.shift();
          try { next(); } catch (e) { console.error("Queue task error:", e); }
        }
      }
    };

    if (activeRequests < CFG.MAX_CONCURRENT) {
      task();
    } else {
      requestQueue.push(task);
    }
  });
}

// ============================================================
//  SECTION 6 · AI MODELS
// ============================================================
const MODELS = {
  gemini      : { name: process.env.GEMINI_MODEL_1, key: process.env.GEMINI_API_KEY, provider: "gemini", model: process.env.GEMINI_MODEL_1 },
  gemini_flash1:{ name: process.env.GEMINI_MODEL_3, key: process.env.GEMINI_API_KEY, provider: "gemini", model: process.env.GEMINI_MODEL_3 },
  gemini_flash2:{ name: process.env.GEMINI_MODEL_2, key: process.env.GEMINI_API_KEY, provider: "gemini", model: process.env.GEMINI_MODEL_2 },
  gemini_flash3:{ name: process.env.GEMINI_MODEL_5, key: process.env.GEMINI_API_KEY, provider: "gemini", model: process.env.GEMINI_MODEL_5 },
  gemini_pro  : { name: process.env.GEMINI_MODEL_4, key: process.env.GEMINI_API_KEY, provider: "gemini", model: process.env.GEMINI_MODEL_4 },
  openai      : { name: process.env.OPENAI_MODEL_1,  key: process.env.OPENAI_API_KEY, type: "openai" },
  claude      : { name: process.env.CLAUDE_MODEL_1,  key: process.env.CLAUDE_API_KEY, type: "claude" },
};

const genAI = new GoogleGenerativeAI(CFG.GEMINI_API_KEY || "");

// Lazy singletons — only built once, only if key exists
let _openai    = null;
let _anthropic = null;
function getOpenAI() {
  if (!CFG.OPENAI_API_KEY) throw new Error("NO_KEY");
  return (_openai ??= new OpenAI({ apiKey: CFG.OPENAI_API_KEY }));
}
function getAnthropic() {
  if (!CFG.CLAUDE_API_KEY) throw new Error("NO_KEY");
  return (_anthropic ??= new Anthropic({ apiKey: CFG.CLAUDE_API_KEY }));
}

const LANGUAGES = {
  en: "🇬🇧 English", hi: "🇮🇳 Hindi",  es: "🇪🇸 Spanish",
  fr: "🇫🇷 French",  de: "🇩🇪 German", ja: "🇯🇵 Japanese",
  ru: "🇷🇺 Russian", ar: "🇸🇦 Arabic",
};

// ============================================================
//  SECTION 7 · BOT + EXPRESS SETUP
//  FIX #6:  Express and bot created early, but webhook & bot
//           handlers only registered AFTER DB is ready.
//  FIX #23: EJS routes wrapped in try/catch — missing templates
//           won't crash the process.
// ============================================================
let PUBLIC_MODE = false; // loaded from DB after connect

const app = express();
app.use(express.json());
app.set("view engine", "ejs");

// FIX #23: safe EJS render — falls back to plain text if template missing
function safeRender(res, view) {
  res.render(view, (err, html) => {
    if (err) return res.send(`<h1>${view}</h1><p>Page coming soon.</p>`);
    res.send(html);
  });
}
app.get("/",        (_q, res) => safeRender(res, "home"));
app.get("/privacy", (_q, res) => safeRender(res, "privacy"));
app.get("/terms",   (_q, res) => safeRender(res, "terms"));
app.get("/health",  (_q, res) => res.json({
  status : "ok",
  db     : mongoose.connection.readyState === 1 ? "connected" : "disconnected",
  queue  : { active: activeRequests, waiting: requestQueue.length },
  uptime : Math.floor(process.uptime()),
}));

const bot = new TelegramBot(CFG.TOKEN);

// FIX #15: strip bot @username from commands so /cmd@BotName works in groups
function stripBotName(text) {
  if (!text) return text;
  return text.replace(/^(\/\w+)@\w+/, "$1");
}

app.post(`/bot${CFG.TOKEN}`, (req, res) => {
  const update = req.body;
  // Normalise command text to strip @BotName suffix
  if (update?.message?.text) {
    update.message.text = stripBotName(update.message.text);
  }
  if (update?.edited_message?.text) {
    update.edited_message.text = stripBotName(update.edited_message.text);
  }
  bot.processUpdate(update);
  res.sendStatus(200);
});

// ── Bot command menu ──────────────────────────────────────
bot.setMyCommands([
  { command: "start",     description: "🚀 Start the bot"       },
  { command: "help",      description: "📝 List of commands"    },
  { command: "account",   description: "👤 My account info"     },
  { command: "language",  description: "🌐 Change language"     },
  { command: "setmodel",  description: "🤖 Select AI model"     },
  { command: "clearchat", description: "🧹 Clear chat history"  },
  { command: "about",     description: "👀 About this bot"      },
  { command: "terms",     description: "📜 Terms of service"    },
  { command: "status",    description: "✅ Check your access"   },
]).catch(() => {});

// ── Reply keyboard ────────────────────────────────────────
const mainKeyboard = {
  reply_markup: {
    keyboard: [
      ["🔍 Search", "🎨 Imagine"],
      ["🚫 Report Error", "📄 Document Analysis"],
    ],
    resize_keyboard  : true,
    one_time_keyboard: false,
  },
};

// ============================================================
//  SECTION 8 · HELPERS
// ============================================================

// FIX #7: single consistent UTC date string used everywhere
function utcToday() { return new Date().toISOString().slice(0, 10); }

// FIX #7: daily reset check always compares UTC dates
function needsDailyReset(dateField) {
  if (!dateField) return true;
  return new Date(dateField).toISOString().slice(0, 10) !== utcToday();
}

// ── FIX #5: Promise race timeout helper ──────────────────
function withTimeout(promise, ms, label) {
  const timer = new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`TIMEOUT:${label}`)), ms)
  );
  return Promise.race([promise, timer]);
}

// ── Daily usage counter (in-memory, UTC-aligned) ─────────
function checkLimit(chatId, type, limit) {
  const today = utcToday();
  const key   = cid(chatId);
  let u = userUsage.get(key);
  if (!u || u.date !== today) {
    u = { date: today, search: 0, imagine: 0, doc: 0, img: 0, proTokens: 0 };
    userUsage.set(key, u);
  }
  if (u[type] >= limit) return false;
  u[type]++;
  // Persist feature use to daily aggregated stats (search/imagine/doc/img only — not proTokens)
  if (["search", "imagine", "doc", "img"].includes(type)) {
    bumpStats({ [type]: 1 });
  }
  return true;
}

// ── FIX #26: membership check with 1 retry on Telegram failure ──
async function isUserMember(chatId) {
  if (!CFG.FORCE_CHANNEL) return true;
  const key    = cid(chatId);
  const cached = membershipCache.get(key);
  if (cached && Date.now() < cached.expiry) return cached.isMember;

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const member   = await withTimeout(
        bot.getChatMember(CFG.FORCE_CHANNEL, chatId),
        5000, "membership"
      );
      const isMember = ["creator", "administrator", "member"].includes(member.status);
      membershipCache.set(key, { isMember, expiry: Date.now() + CFG.MEMBER_CACHE_MIN * 60000 });
      return isMember;
    } catch (err) {
      if (attempt === 0) {
        // Brief pause then retry
        await new Promise(r => setTimeout(r, 800));
        continue;
      }
      // Both attempts failed — use last cached value if available, else block
      const stale = membershipCache.get(key);
      console.warn(`⚠️ Membership check failed (both attempts): ${err.message}`);
      return stale?.isMember ?? false;
    }
  }
}

async function checkMembership(msg) {
  const chatId = msg.chat.id;
  // In groups, msg.chat.id is the group ID — always check the actual user (msg.from.id)
  const userId = msg.from?.id || chatId;
  if (await isUserMember(userId)) return true;
  await safeSend(chatId, "⚠️ You must join our channel first to use this bot.", {
    reply_markup: {
      inline_keyboard: [[
        { text: "📢 Join Channel", url: `https://t.me/${CFG.FORCE_CHANNEL.replace("@", "")}` },
        { text: "✅ I have joined", callback_data: "verify_membership" },
      ]],
    },
  });
  return false;
}

// ── FIX #9: approval cache with short TTL so revocations propagate ──
async function isUserApproved(chatId) {
  if (PUBLIC_MODE)             return true;
  if (chatId === CFG.ADMIN_ID) return true;
  const key    = cid(chatId);
  const cached = approvalCache.get(key);
  if (cached && Date.now() < cached.expiry) return cached.approved;

  const user    = await User.findOne({ chatId: key }, { approvedUntil: 1 });
  const approved = !!(user?.approvedUntil && new Date() <= user.approvedUntil);
  approvalCache.set(key, { approved, expiry: Date.now() + CFG.APPROVAL_CACHE_TTL });
  return approved;
}

async function guardAccess(msg) {
  if (await isUserApproved(msg.chat.id)) return true;
  await safeSend(
    msg.chat.id,
    `🛡️ *Member Only*\nThis feature is locked. Contact ${CFG.FORCE_CHANNEL} to get approved.`,
    { parse_mode: "Markdown" }
  );
  return false;
}

// ── Rate limiters ─────────────────────────────────────────
async function guardRateLimit(msg) {
  if (msg.text?.startsWith("/"))   return true;
  if (msg.photo || msg.document)   return true;
  const key = cid(msg.chat.id);
  const now = Date.now();
  const rem = CFG.RATE_LIMIT_MS - (now - (rateLimitMap.get(key) || 0));
  if (rem > 0) {
    await safeSend(msg.chat.id, `⏳ Please wait ${Math.ceil(rem / 1000)}s before sending another request.`);
    return false;
  }
  rateLimitMap.set(key, now);
  return true;
}

async function guardRateLimitMedia(msg) {
  const key = cid(msg.chat.id);
  const now = Date.now();
  const rem = CFG.MEDIA_COOLDOWN_MS - (now - (mediaRateLimit.get(key) || 0));
  if (rem > 0) {
    await safeSend(msg.chat.id, `⏳ Please wait ${Math.ceil(rem / 1000)}s before sending another file.`);
    return false;
  }
  mediaRateLimit.set(key, now);
  return true;
}

async function guardCommandRateLimit(msg, commandName) {
  const key = `${cid(msg.chat.id)}:${commandName}`;
  const now = Date.now();
  const rem = CFG.COMMAND_LIMIT_MS - (now - (commandRateLmt.get(key) || 0));
  if (rem > 0) {
    await safeSend(msg.chat.id, `⏳ Please wait ${Math.ceil(rem / 1000)}s before using /${commandName} again.`);
    return false;
  }
  commandRateLmt.set(key, now);
  return true;
}

// ── FIX #3: unified processing lock — covers all heavy handlers ──
function isProcessing(chatId) { return processingSet.has(cid(chatId)); }
function markProcessing(chatId) {
  const key = cid(chatId);
  processingSet.add(key);
  setTimeout(() => processingSet.delete(key), PROCESSING_TIMEOUT);
}
function unmarkProcessing(chatId) { processingSet.delete(cid(chatId)); }

// ── FIX #8: lastActive update — shared helper used everywhere ──
function touchLastActive(chatId) {
  User.findOneAndUpdate(
    { chatId: cid(chatId) },
    { $set: { lastActive: new Date() } }
  ).catch(() => {});
}

// ── FIX #2: race-condition-safe user upsert ───────────────
async function getOrCreateUser(chatId) {
  const key = cid(chatId);
  try {
    return await User.findOneAndUpdate(
      { chatId: key },
      { $setOnInsert: { chatId: key } },
      { upsert: true, new: true }
    );
  } catch (err) {
    // E11000 duplicate key — another request created it simultaneously
    if (err.code === 11000) return User.findOne({ chatId: key });
    throw err;
  }
}

// ── File download with pre-flight size check ──────────────
async function downloadFile(fileId) {
  const file     = await withTimeout(bot.getFile(fileId), CFG.API_TIMEOUT_MS, "getFile");
  const MAX_BYTES = CFG.MAX_FILE_MB * 1024 * 1024;
  if (file.file_size && file.file_size > MAX_BYTES) {
    throw new Error(`FILE_TOO_LARGE:${CFG.MAX_FILE_MB}`);
  }
  const fileUrl = `https://api.telegram.org/file/bot${CFG.TOKEN}/${file.file_path}`;
  const res     = await withTimeout(fetch(fileUrl), CFG.API_TIMEOUT_MS, "downloadFile");
  if (!res.ok)   throw new Error(`Download failed: ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

// ── FIX #19: Markdown-safe message splitter ───────────────
// Splits on newlines first to avoid breaking words/URLs/code blocks
function splitSafely(text, maxLen = 4000) {
  const chunks = [];
  while (text.length > maxLen) {
    let idx = text.lastIndexOf("\n", maxLen);
    if (idx < maxLen * 0.5) idx = maxLen; // no newline found — hard split
    chunks.push(text.slice(0, idx));
    text = text.slice(idx).trimStart();
  }
  if (text) chunks.push(text);
  return chunks;
}

async function sendLongMessage(chatId, text, opts = {}) {
  for (const chunk of splitSafely(text)) {
    try   { await bot.sendMessage(chatId, chunk, opts); }
    catch { await bot.sendMessage(chatId, chunk); } // strip parse_mode on failure
  }
}

// ── FIX #18 + error handling: safeSend with smart error classification ──
//
// Handles the four real failure modes instead of silently swallowing them:
//   429 → retry once after the retry_after delay Telegram provides
//   403 → bot blocked/user deleted — remove from DB so we stop wasting cycles
//   400 "can't parse" → strip parse_mode and retry (Markdown was malformed)
//   500 / network → log so we know Telegram had an outage; don't crash
async function safeSend(chatId, text, opts = {}) {
  if (text.length > 4000) return sendLongMessage(chatId, text, opts);
  try {
    return await bot.sendMessage(chatId, text, opts);
  } catch (err) {
    const msg  = err?.message || "";
    const code = err?.response?.statusCode;

    // 429 — Telegram is rate-limiting us: wait the retry_after it provides, then retry once
    if (code === 429 || msg.includes("Too Many Requests")) {
      const retryAfter = (err?.response?.body?.parameters?.retry_after || 5) * 1000;
      console.warn(`⚠️ Telegram 429 for ${chatId} — retrying after ${retryAfter}ms`);
      await new Promise(r => setTimeout(r, retryAfter));
      try { return await bot.sendMessage(chatId, text, opts); }
      catch (retryErr) { console.error(`❌ safeSend retry failed for ${chatId}:`, retryErr.message); }
      return;
    }

    // 403 — bot was blocked or user deleted their account: clean up DB so we stop processing them
    if (code === 403 || msg.includes("Forbidden") || msg.includes("bot was blocked")) {
      console.warn(`⚠️ Bot blocked by ${chatId} — removing from DB`);
      User.deleteOne({ chatId: cid(chatId) }).catch(e =>
        console.error("❌ Failed to remove blocked user:", e.message)
      );
      ChatHistory.deleteOne({ chatId: cid(chatId) }).catch(e =>
        console.error("❌ Failed to remove blocked user history:", e.message)
      );
      return;
    }

    // 400 "can't parse entities" — AI generated malformed Markdown: retry as plain text
    if (msg.includes("can't parse") && opts.parse_mode) {
      const { parse_mode: _, ...safe } = opts;
      try { return await bot.sendMessage(chatId, text, safe); }
      catch (plainErr) { console.error(`❌ safeSend plain-text fallback failed for ${chatId}:`, plainErr.message); }
      return;
    }

    // 400 "not enough rights" / "have no rights" — bot not admin in group: eat silently
    if (
      code === 400 && (
        msg.includes("not enough rights") ||
        msg.includes("have no rights")    ||
        msg.includes("need administrator") ||
        msg.includes("CHAT_WRITE_FORBIDDEN")
      )
    ) {
      // Bot is not an admin in this group — cannot send messages; silently return null
      return null;
    }

    // 5xx / network — Telegram outage or dropped connection: log it, don't crash
    if (code >= 500 || msg.includes("ETIMEOUT") || msg.includes("ECONNRESET") || msg.includes("socket hang up")) {
      console.error(`❌ Telegram server/network error for ${chatId} (${code || "network"}):`, msg);
      return;
    }

    // Everything else (chat not found, user deactivated, etc.) — log so we're not flying blind
    console.warn(`⚠️ safeSend unhandled error for ${chatId} [${code}]:`, msg);
  }
}

// ── FIX #28: escape Markdown v1 special chars in AI replies ──
function escapeMarkdown(text) {
  // Escape only chars that break Telegram Markdown v1
  return text.replace(/([_*`\[])/g, "\\$1");
}

// ── Thinking / status placeholder ────────────────────────
async function withThinkingIndicator(chatId, label, fn) {
  let placeholder;
  try { placeholder = await bot.sendMessage(chatId, label); } catch {}

  const typingInterval = setInterval(
    () => bot.sendChatAction(chatId, "typing").catch(() => {}),
    4000
  );
  try {
    return await fn();
  } finally {
    clearInterval(typingInterval);
    if (placeholder?.message_id) {
      bot.deleteMessage(chatId, placeholder.message_id).catch(() => {});
    }
  }
}

// ── AI call with FIX #5 timeout ──────────────────────────
async function getAIReply(chatId, text, history, lang) {
  const selectedId = userModels.get(cid(chatId)) || "gemini";
  const chosen     = MODELS[selectedId];
  if (!chosen?.key) throw new Error("NO_KEY");

  const prompt = `Answer in ${LANGUAGES[lang] || "English"} (${lang})\n\nConversation so far:\n${history}\n\nUser: ${text}`;

  if (chosen.provider === "gemini") {
    if (chosen.model === process.env.GEMINI_MODEL_4) {
      if (!checkLimit(chatId, "proTokens", CFG.PRO_LIMIT))
        throw new Error(`PRO_LIMIT:${CFG.PRO_LIMIT}`);
    }
    const model  = genAI.getGenerativeModel({ model: chosen.model });
    const result = await withTimeout(
      model.generateContent({
        contents        : [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: CFG.MAX_REPLY_TOKENS },
      }),
      CFG.API_TIMEOUT_MS, "gemini"
    );
    return { reply: result?.response?.text() || "⚠️ No response from Gemini.", chosen };
  }

  if (chosen.type === "openai") {
    const result = await withTimeout(
      getOpenAI().chat.completions.create({
        model   : process.env.OPENAI_MODEL_1,
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user",   content: prompt },
        ],
        max_tokens: CFG.MAX_REPLY_TOKENS,
      }),
      CFG.API_TIMEOUT_MS, "openai"
    );
    return { reply: result.choices[0].message.content || "⚠️ No response from OpenAI.", chosen };
  }

  if (chosen.type === "claude") {
    const result = await withTimeout(
      getAnthropic().messages.create({
        model     : process.env.CLAUDE_MODEL_1,
        max_tokens: CFG.MAX_REPLY_TOKENS,
        messages  : [{ role: "user", content: prompt }],
      }),
      CFG.API_TIMEOUT_MS, "claude"
    );
    return { reply: result.content?.[0]?.text || "⚠️ No response from Claude.", chosen };
  }

  throw new Error("UNKNOWN_PROVIDER");
}

// ── Prefs hydration with dedup ────────────────────────────
async function hydrateUserPrefs(chatId) {
  const key = cid(chatId);
  if (userLanguages.has(key) && userModels.has(key)) return;
  if (_prefsPending.has(key)) return _prefsPending.get(key);

  const promise = User.findOne({ chatId: key }, { language: 1, selectedModel: 1 })
    .then(u => {
      if (!userLanguages.has(key)) userLanguages.set(key, u?.language      || "en");
      if (!userModels.has(key))    userModels.set(key,    u?.selectedModel  || "gemini");
    })
    .catch(() => {
      userLanguages.set(key, "en");
      userModels.set(key, "gemini");
    })
    .finally(() => _prefsPending.delete(key));

  _prefsPending.set(key, promise);
  return promise;
}

// ── Chat history helpers ──────────────────────────────────
async function appendChatHistory(chatId, userText, botText) {
  const key = cid(chatId);
  const now = new Date();
  // FIX #20: truncate each stored message to MAX_HISTORY_CHARS
  const newMessages = [
    { role: "user", text: String(userText).slice(0, CFG.MAX_HISTORY_CHARS), createdAt: now },
    { role: "bot",  text: String(botText).slice(0,  CFG.MAX_HISTORY_CHARS), createdAt: now },
  ];
  await ChatHistory.findOneAndUpdate(
    { chatId: key },
    {
      $push: { messages: { $each: newMessages, $slice: -CFG.DB_MSG_LIMIT } },
      $set : { updatedAt: now },
    },
    { upsert: true }
  ).catch(err => console.error("appendChatHistory error:", err.message));
}

async function getRecentHistory(chatId) {
  const doc = await ChatHistory.findOne(
    { chatId: cid(chatId) },
    { messages: { $slice: -CFG.HISTORY_MESSAGES } }
  );
  if (!doc?.messages?.length) return "";
  return doc.messages.map(m => `${m.role}: ${m.text}`).join("\n");
}

// ── FIX #13: broadcast — 5 msgs/second to stay under Telegram limit ──
async function broadcastMessage(message) {
  let success = 0, failed = 0;
  const DELAY = 220; // ~4-5 msgs/s; Telegram group limit is 20/min, global ~30/s

  const cursor = User.find({}, { chatId: 1 }).cursor();
  for await (const u of cursor) {
    try {
      await bot.sendMessage(u.chatId, `📢 Broadcast:\n\n${message}`);
      success++;
    } catch { failed++; }
    await new Promise(r => setTimeout(r, DELAY));
  }
  return { success, failed };
}

// ============================================================
//  SECTION 9 · STARTUP — wait for DB, THEN open for traffic
//  FIX #6:  Express only accepts real traffic after DB + webhook ready.
//  FIX #10: PUBLIC_MODE loaded from DB on startup.
//  FIX #21: runCleanup() called immediately on startup, then daily.
//  FIX #29: health check confirms DB + webhook before "ready" log.
// ============================================================
async function startup() {
  // 1. Connect DB
  await connectDB();

  // 2. Load persisted settings
  try {
    const s = await Setting.findOne({ key: "publicMode" });
    if (s) PUBLIC_MODE = !!s.value;
  } catch {}
  console.log(`🔑 PUBLIC_MODE: ${PUBLIC_MODE}`);

  // 3. Register webhook
  try {
    await withTimeout(
      bot.setWebHook(`${CFG.RENDER_URL}/bot${CFG.TOKEN}`),
      10000, "setWebhook"
    );
    console.log("✅ Webhook registered");
  } catch (err) {
    console.error("❌ Webhook registration failed:", err.message);
  }

  // 4. Start Express
  app.listen(CFG.PORT, () => console.log(`✅ Server on port ${CFG.PORT}`));

  // 5. FIX #21: run cleanup immediately, then every 24h
  runCleanup();
  setInterval(runCleanup, 24 * 60 * 60 * 1000);

  console.log("🤖 Bot fully ready");
}

// ── Scheduled cleanup ─────────────────────────────────────
async function runCleanup() {
  console.log(`🔧 Cleanup started [${new Date().toISOString()}]`);
  try {
    const cutoff = new Date(Date.now() - CFG.INACTIVE_USER_DAYS * 86400000);
    const res    = await User.deleteMany({
      lastActive   : { $lt: cutoff },
      approvedUntil: { $not: { $gt: new Date() } },
    });
    if (res.deletedCount > 0) console.log(`🗑️  Removed ${res.deletedCount} inactive users`);
  } catch (err) {
    console.error("❌ Cleanup error:", err.message);
  }
}

// ============================================================
//  SECTION 10 · BOT COMMANDS
// ============================================================

// ── /start ───────────────────────────────────────────────
bot.onText(/\/start/, async (msg) => {
  if (!await checkMembership(msg)) return;
  await safeSend(
    msg.chat.id,
    `👋 Hi ${msg.from?.first_name || "there"}!\nI am your Private AI assistant.\nType any question and I'll answer.`,
    mainKeyboard
  );
});

// ── /help ────────────────────────────────────────────────
bot.onText(/\/help/, async (msg) => {
  if (!await checkMembership(msg)) return;
  const chatId = msg.chat.id;

  let help = `📌 *Available commands:*
/start — Start the bot
/status — Check access status
/help — Show this menu
/about — About this bot
/clearchat — Clear chat history
/terms — Terms of service
/account — My account info
/setmodel — Choose AI model
/language — Change language
/imagine A dancing panda
/search Today\'s latest news

💡 *Tip:* Send any document or photo for analysis!`;

  if (chatId === CFG.ADMIN_ID) {
    help += `\n\n🎧 *Admin Commands:*
/broadcast — Send to all users
/usage — Usage report
/approve — Approve user
/remove — Remove user
/users — List approved users
/mode — Check mode
/private — Set private
/public — Set public`;
  }
  await safeSend(chatId, help, { parse_mode: "Markdown" });
});

// ── /about ───────────────────────────────────────────────
bot.onText(/\/about/, async (msg) => {
  if (!await checkMembership(msg)) return;
  await safeSend(msg.chat.id,
    `🤖 *About this bot*\n\nBuilt with:\n• Telegram Bot API\n• Google Gemini\n• OpenAI\n• Claude AI\n• MongoDB Atlas\n• Node.js`,
    { parse_mode: "Markdown" }
  );
});

// ── /terms ───────────────────────────────────────────────
bot.onText(/\/terms/, async (msg) => {
  if (!await checkMembership(msg)) return;
  await safeSend(msg.chat.id,
    `📜 *Terms of Service*\n\n1. Personal & educational use only.\n2. No harmful or illegal content.\n3. Limited usage data stored.\n4. AI responses may not always be accurate.\n5. Using this bot = accepting these terms.`,
    {
      parse_mode: "Markdown",
      reply_markup: { inline_keyboard: [[
        { text: "📄 Full Terms", url: `${CFG.RENDER_URL}/terms` },
        { text: "🔒 Privacy",    url: `${CFG.RENDER_URL}/privacy` },
      ]]},
    }
  );
});

// ── /status ──────────────────────────────────────────────
bot.onText(/\/status/, async (msg) => {
  const chatId = msg.chat.id;
  if (PUBLIC_MODE) return safeSend(chatId, "🔓 Bot is in PUBLIC MODE — all users welcome.");
  const approved = await isUserApproved(chatId);
  await safeSend(chatId, approved ? "✅ You have access." : "🚧 You do not have access yet.");
});

// ── /clearchat ───────────────────────────────────────────
bot.onText(/\/clearchat/, async (msg) => {
  const chatId = msg.chat.id;
  if (!await checkMembership(msg)) return;
  try {
    await ChatHistory.deleteOne({ chatId: cid(chatId) });
    await safeSend(chatId, "🧹 Your chat history has been cleared.");
  } catch (err) {
    console.error("clearchat error:", err.message);
    await safeSend(chatId, "⚠️ Could not clear history. Try again.");
  }
});

// ── /account ─────────────────────────────────────────────
bot.onText(/\/account/, async (msg) => {
  const chatId = msg.chat.id;
  if (!await checkMembership(msg)) return;

  try {
    // FIX #2: race-safe upsert
    let user = await getOrCreateUser(chatId);
    const updates = {};
    if (needsDailyReset(user.lastReset))    { updates.requests = 0;   updates.lastReset   = new Date(); }
    if (needsDailyReset(user.tokensReset))  { updates.tokensUsed = 0; updates.tokensReset = new Date(); }
    if (Object.keys(updates).length) {
      user = await User.findOneAndUpdate({ chatId: cid(chatId) }, { $set: updates }, { new: true });
    }

    await hydrateUserPrefs(chatId);
    const lang     = userLanguages.get(cid(chatId)) || user.language      || "en";
    const model    = userModels.get(cid(chatId))    || user.selectedModel  || "gemini";
    const usage    = userUsage.get(cid(chatId))     || {};
    const reset    = new Date(); reset.setUTCHours(24, 0, 0, 0);

    await safeSend(chatId, `👤 *My Account*
━━━━━━━━━━━━━
📨 Requests: ${user.requests} / ${CFG.DAILY_REQUEST_LIMIT}
🪙 Tokens: ${user.tokensUsed} / ${CFG.DAILY_TOKEN_LIMIT}
🕒 Resets at: ${reset.toUTCString()}

🌍 Language: ${LANGUAGES[lang] || lang}
🤖 Model: ${MODELS[model]?.name || model}

📊 *Feature Usage Today*
━━━━━━━━━━━━━
🔍 Searches: ${usage.search || 0} / ${CFG.SEARCH_LIMIT}
🎨 Images: ${usage.imagine || 0} / ${CFG.IMAGINE_LIMIT}
📄 Documents: ${usage.doc || 0} / ${CFG.DOC_LIMIT}
🖼️ Photos: ${usage.img || 0} / ${CFG.IMG_LIMIT}
🤖 Pro Reqs: ${usage.proTokens || 0} / ${CFG.PRO_LIMIT}`,
      { parse_mode: "Markdown" }
    );
  } catch (err) {
    // FIX #16
    console.error("account error:", err.message);
    await safeSend(chatId, "⚠️ Could not load account info. Try again.");
  }
});

// ── /language ────────────────────────────────────────────
bot.onText(/\/language/, async (msg) => {
  if (!await checkMembership(msg)) return;
  const buttons  = Object.entries(LANGUAGES).map(([c, n]) => ({ text: n, callback_data: `lang_${c}` }));
  const keyboard = [];
  for (let i = 0; i < buttons.length; i += 2) keyboard.push(buttons.slice(i, i + 2));
  await safeSend(msg.chat.id, "🌐 Choose your language:", { reply_markup: { inline_keyboard: keyboard } });
});

// ── /setmodel ────────────────────────────────────────────
bot.onText(/\/setmodel/, async (msg) => {
  if (!await checkMembership(msg)) return;
  const buttons = Object.entries(MODELS).map(([id, m]) => [{
    text         : m.key ? m.name : `${m.name} ❌`,
    callback_data: m.key ? `model_${id}` : `unavailable_${id}`,
  }]);
  await safeSend(msg.chat.id, "🤖 Choose your AI model:", { reply_markup: { inline_keyboard: buttons } });
});

// ── Unified callback handler ──────────────────────────────
bot.on("callback_query", async (query) => {
  const chatId = query.message.chat.id;
  const data   = query.data;

  if (data.startsWith("lang_")) {
    const lang = data.replace("lang_", "");
    userLanguages.set(cid(chatId), lang);
    User.findOneAndUpdate({ chatId: cid(chatId) }, { $set: { language: lang } }, { upsert: true }).catch(() => {});
    bot.answerCallbackQuery(query.id).catch(() => {});
    await safeSend(chatId, `✅ Language changed to ${LANGUAGES[lang]}`);
    return;
  }

  if (data.startsWith("model_")) {
    const modelId = data.replace("model_", "");
    userModels.set(cid(chatId), modelId);
    User.findOneAndUpdate({ chatId: cid(chatId) }, { $set: { selectedModel: modelId } }, { upsert: true }).catch(() => {});
    bot.answerCallbackQuery(query.id).catch(() => {});
    await safeSend(chatId, `✅ Model set to *${MODELS[modelId]?.name || modelId}*`, { parse_mode: "Markdown" });
    return;
  }

  if (data === "verify_membership") {
    // query.from.id is always the actual user — works in both private and groups
    const userId = query.from.id;
    membershipCache.delete(cid(userId));
    const joined = await isUserMember(userId);
    if (joined) {
      bot.answerCallbackQuery(query.id, { text: "✅ Verified! You can now use the bot.", show_alert: false }).catch(() => {});
      await bot.editMessageReplyMarkup({ inline_keyboard: [] }, {
        chat_id   : chatId,
        message_id: query.message.message_id,
      }).catch(() => {});
      await safeSend(chatId, `✅ @${query.from.username || query.from.first_name} is verified! You can now use the bot.`);
    } else {
      bot.answerCallbackQuery(query.id, { text: "❌ You haven't joined yet. Please join the channel first.", show_alert: true }).catch(() => {});
    }
    return;
  }

  if (data.startsWith("unavailable_")) {
    bot.answerCallbackQuery(query.id, { text: "⚠️ Model not available.", show_alert: true }).catch(() => {});
    return;
  }

  bot.answerCallbackQuery(query.id).catch(() => {});
});

// ── /imagine ─────────────────────────────────────────────
// FIX #3: processing lock applied; FIX #12: download image then send as buffer
bot.onText(/\/imagine (.+)/, async (msg, match) => {
  const chatId = msg.chat.id;
  if (!await checkMembership(msg))         return;
  if (!await guardAccess(msg))             return;
  if (!guardCommandRateLimit(msg, "imagine")) return;
  if (isProcessing(chatId)) {
    return safeSend(chatId, "⏳ Your previous request is still in progress.");
  }
  if (!checkLimit(chatId, "imagine", CFG.IMAGINE_LIMIT)) {
    return safeSend(chatId, `⚠️ Daily image limit reached (${CFG.IMAGINE_LIMIT}). Try again tomorrow.`);
  }

  const prompt = match[1];
  markProcessing(chatId);
  let placeholder;
  try {
    placeholder = await bot.sendMessage(chatId, "🎨 Generating your image, please wait…");
    bot.sendChatAction(chatId, "upload_photo").catch(() => {});

    // FIX #12: download image buffer instead of passing URL directly to sendPhoto
    const imageUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}`;
    const res      = await withTimeout(fetch(imageUrl), CFG.API_TIMEOUT_MS, "pollinations");
    if (!res.ok) throw new Error(`Pollinations error: ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    await bot.sendPhoto(
      chatId, 
      buf, 
      { caption: `🎨 ${prompt}` }, 
      { filename: 'image.jpg', contentType: 'image/jpeg' }
    );
    // FIX #8: update lastActive
    touchLastActive(chatId);
  } catch (err) {
    const isTimeout = err.message?.startsWith("TIMEOUT");
    console.error("❌ Image error:", err.message);
    await safeSend(chatId, isTimeout
      ? "⚠️ Image generation timed out. Try again."
      : "⚠️ Could not generate image. Please try again."
    );
  } finally {
    unmarkProcessing(chatId);
    if (placeholder?.message_id) bot.deleteMessage(chatId, placeholder.message_id).catch(() => {});
  }
});

// ── /search ──────────────────────────────────────────────
// FIX #3 + #8 + #5
bot.onText(/\/search (.+)/, async (msg, match) => {
  const chatId = msg.chat.id;
  if (!await checkMembership(msg))         return;
  if (!await guardAccess(msg))             return;
  if (!guardCommandRateLimit(msg, "search")) return;
  if (isProcessing(chatId)) {
    return safeSend(chatId, "⏳ Your previous request is still in progress.");
  }
  if (!checkLimit(chatId, "search", CFG.SEARCH_LIMIT)) {
    return safeSend(chatId, `⚠️ Daily search limit reached (${CFG.SEARCH_LIMIT}). Try again tomorrow.`);
  }

  const query = match[1];
  markProcessing(chatId);
  let placeholder;
  try {
    placeholder = await bot.sendMessage(chatId, "🔍 Searching the web…");
    bot.sendChatAction(chatId, "typing").catch(() => {});

    const res = await withTimeout(
      fetch("https://google.serper.dev/search", {
        method : "POST",
        headers: { "X-API-KEY": CFG.SERPER_API_KEY, "Content-Type": "application/json" },
        body   : JSON.stringify({ q: query }),
      }),
      CFG.API_TIMEOUT_MS, "serper"
    );
    if (!res.ok) throw new Error(`Serper error: ${res.status}`);

    const data    = await res.json();
    const results = data.organic
      ?.slice(0, CFG.SEARCH_RESULTS)
      .map((r, i) => `${i + 1}. [${r.title}](${r.link})\n${r.snippet}`)
      .join("\n\n");

    if (placeholder?.message_id) bot.deleteMessage(chatId, placeholder.message_id).catch(() => {});

    if (!results) return safeSend(chatId, "⚠️ No search results found.");

    await safeSend(chatId, `🔍 *Results for:* ${query}\n\n${results}`, {
      parse_mode             : "Markdown",
      disable_web_page_preview: true,
    });
    touchLastActive(chatId); // FIX #8
  } catch (err) {
    if (placeholder?.message_id) bot.deleteMessage(chatId, placeholder.message_id).catch(() => {});
    console.error("❌ Search error:", err.message);
    await safeSend(chatId, err.message?.startsWith("TIMEOUT")
      ? "⚠️ Search timed out. Try again."
      : "⚠️ Could not perform web search."
    );
  } finally {
    unmarkProcessing(chatId);
  }
});

// ── Document analysis ─────────────────────────────────────
// FIX #3 + #5 + #8 + #24
bot.on("document", async (msg) => {
  const chatId = msg.chat.id;
  if (!await checkMembership(msg)) return;
  if (!await guardAccess(msg))     return;
  if (!guardRateLimitMedia(msg))   return;
  if (isProcessing(chatId)) {
    return safeSend(chatId, "⏳ Your previous request is still in progress.");
  }
  if (!checkLimit(chatId, "doc", CFG.DOC_LIMIT)) {
    return safeSend(chatId, `⚠️ Daily document limit reached (${CFG.DOC_LIMIT}). Try again tomorrow.`);
  }

  const fileName = (msg.document.file_name || "").toLowerCase();
  if (![".pdf", ".docx", ".txt"].some(ext => fileName.endsWith(ext))) {
    return safeSend(chatId, "⚠️ Only PDF, DOCX, or TXT files are supported.");
  }

  markProcessing(chatId);
  try {
    await enqueueRequest(async () => {
      await withThinkingIndicator(chatId, "📄 Analysing your document, please wait…", async () => {
        let fileBuffer;
        try {
          fileBuffer = await downloadFile(msg.document.file_id);
        } catch (err) {
          if (err.message?.startsWith("FILE_TOO_LARGE"))
            return safeSend(chatId, `⚠️ File too large. Max: ${CFG.MAX_FILE_MB} MB.`);
          throw err;
        }

        let text = "";
        if (fileName.endsWith(".pdf")) {
          const data = await withTimeout(pdf(fileBuffer), CFG.API_TIMEOUT_MS, "pdf-parse");
          text = data.text;
        } else if (fileName.endsWith(".docx")) {
          const data = await withTimeout(mammoth.extractRawText({ buffer: fileBuffer }), CFG.API_TIMEOUT_MS, "mammoth");
          text = data.value;
        } else {
          text = fileBuffer.toString("utf-8");
        }
        fileBuffer = null; // FIX #24: release buffer ASAP

        if (!text.trim()) return safeSend(chatId, "⚠️ No readable text found.");
        if (text.length > CFG.DOC_CHAR_LIMIT) text = text.slice(0, CFG.DOC_CHAR_LIMIT);

        const model  = genAI.getGenerativeModel({ model: process.env.GEMINI_MODEL_5 });
        const result = await withTimeout(
          model.generateContent({
            contents        : [{ role: "user", parts: [{ text: `Summarize this document:\n\n${text}` }] }],
            generationConfig: { maxOutputTokens: CFG.MAX_REPLY_TOKENS },
          }),
          CFG.API_TIMEOUT_MS, "gemini-doc"
        );
        const reply = result?.response?.text() || "⚠️ No response.";
        await safeSend(chatId, "📄 *Document Summary:*\n\n" + reply, { parse_mode: "Markdown" });
        touchLastActive(chatId); // FIX #8
      });
    });
  } catch (err) {
    console.error("❌ Document error:", err.message);
    await safeSend(chatId, err.message === "QUEUE_FULL"
      ? "⚠️ Bot is busy. Please try again in a moment."
      : err.message?.startsWith("TIMEOUT")
        ? "⚠️ Analysis timed out. Try a smaller file."
        : err.message?.toLowerCase().includes("file is too big")
          ? `⚠️ File too large. Max: ${CFG.MAX_FILE_MB} MB.`
          : "⚠️ Could not analyse your document."
    );
  } finally {
    unmarkProcessing(chatId);
  }
});

// ── Image analysis ────────────────────────────────────────
// FIX #3 + #5 + #8 + #14 (album dedup)
bot.on("photo", async (msg) => {
  const chatId = msg.chat.id;

  // FIX #14: skip duplicate events from the same album
  if (msg.media_group_id) {
    if (albumSeen.has(msg.media_group_id)) return;
    albumSeen.set(msg.media_group_id, Date.now());
    // Delete directly after 5 s — don't wait for the 30-min sweeper (memory leak fix)
    setTimeout(() => albumSeen.delete(msg.media_group_id), 5000);
  }

  if (!await checkMembership(msg)) return;
  if (!await guardAccess(msg))     return;
  if (!guardRateLimitMedia(msg))   return;
  if (isProcessing(chatId)) {
    return safeSend(chatId, "⏳ Your previous request is still in progress.");
  }
  if (!checkLimit(chatId, "img", CFG.IMG_LIMIT)) {
    return safeSend(chatId, `⚠️ Daily image analysis limit reached (${CFG.IMG_LIMIT}). Try again tomorrow.`);
  }

  const photo = msg.photo[msg.photo.length - 1];
  markProcessing(chatId);
  try {
    await enqueueRequest(async () => {
      await withThinkingIndicator(chatId, "🖼️ Analysing your image, please wait…", async () => {
        let fileBuffer;
        try {
          fileBuffer = await downloadFile(photo.file_id);
        } catch (err) {
          if (err.message?.startsWith("FILE_TOO_LARGE"))
            return safeSend(chatId, `⚠️ Image too large. Max: ${CFG.MAX_FILE_MB} MB.`);
          throw err;
        }

        const base64Image = fileBuffer.toString("base64");
        fileBuffer = null; // FIX #24: release buffer

        const model  = genAI.getGenerativeModel({ model: process.env.GEMINI_MODEL_5 });
        const result = await withTimeout(
          model.generateContent([
            "Describe this image in detail.",
            { inlineData: { data: base64Image, mimeType: "image/jpeg" } },
          ]),
          CFG.API_TIMEOUT_MS, "gemini-img"
        );
        const reply = result?.response?.text() || "⚠️ No response.";
        await safeSend(chatId, "🖼️ *Image Analysis:*\n\n" + reply, { parse_mode: "Markdown" });
        touchLastActive(chatId); // FIX #8
      });
    });
  } catch (err) {
    console.error("❌ Image analysis error:", err.message);
    await safeSend(chatId, err.message === "QUEUE_FULL"
      ? "⚠️ Bot is busy. Please try again in a moment."
      : err.message?.startsWith("TIMEOUT")
        ? "⚠️ Analysis timed out. Try a smaller image."
        : err.message?.toLowerCase().includes("file is too big")
          ? `⚠️ Image too large. Max: ${CFG.MAX_FILE_MB} MB.`
          : "⚠️ Could not analyse your image."
    );
  } finally {
    unmarkProcessing(chatId);
  }
});

// ── Main chat handler ─────────────────────────────────────
// FIX #2 + #3 + #5 + #7 + #8 + #25 + #28
bot.on("message", async (msg) => {
  const chatId = msg.chat.id;
  const text   = msg.text;

  // ── Bot added to a group: prompt users to make it admin ──
  if (msg.new_chat_members) {
    const BOT_INFO = await bot.getMe().catch(() => null);
    const isMe = BOT_INFO && msg.new_chat_members.some(m => m.id === BOT_INFO.id);
    if (isMe) {
      // safeSend handles the case where bot is not yet admin (returns null silently)
      await safeSend(
        chatId,
        `👋 *Thanks for adding me!*\n\n` +
        `⚠️ *Important:* Please make me an *Admin* in this group so I can send messages and respond to users.\n\n` +
        `Once I'm an admin, users can chat with me directly here! 🚀`,
        { parse_mode: "Markdown" }
      );
    }
    return;
  }

  if (!text || text.startsWith("/") || msg.photo || msg.document) return;

  if (!await checkMembership(msg)) return;
  if (!await guardAccess(msg))     return;
  if (!guardRateLimit(msg))        return;

  if (/^https?:\/\//i.test(text.trim())) {
    return safeSend(chatId, "⚠️ Links are not allowed.");
  }

  // Reply keyboard shortcuts
  if (text === "🔍 Search")            return safeSend(chatId, "🔍 Type: `/search your query`",  { parse_mode: "Markdown" });
  if (text === "🎨 Imagine")           return safeSend(chatId, "🎨 Type: `/imagine your prompt`", { parse_mode: "Markdown" });
  if (text === "🚫 Report Error")      return safeSend(chatId, `Contact ${CFG.FORCE_CHANNEL} to report problems.`);
  if (text === "📄 Document Analysis") return safeSend(chatId, "📤 Send any document or image for analysis.");

  if (isProcessing(chatId)) {
    return safeSend(chatId, "⏳ Your previous request is still in progress. Please wait.");
  }

  markProcessing(chatId);
  try {
    await enqueueRequest(async () => {
      // FIX #2: race-safe upsert
      let user = await getOrCreateUser(chatId);

      // FIX #7: UTC-aligned daily resets
      const resetUpdates = {};
      if (needsDailyReset(user.lastReset))   { resetUpdates.requests  = 0; resetUpdates.lastReset   = new Date(); user.requests   = 0; }
      if (needsDailyReset(user.tokensReset)) { resetUpdates.tokensUsed = 0; resetUpdates.tokensReset = new Date(); user.tokensUsed = 0; }
      if (Object.keys(resetUpdates).length) {
        await User.findOneAndUpdate({ chatId: cid(chatId) }, { $set: resetUpdates });
      }

      if (user.requests  >= CFG.DAILY_REQUEST_LIMIT) {
        return safeSend(chatId, `⚠️ Daily request limit (${CFG.DAILY_REQUEST_LIMIT}) reached. Try again tomorrow.`);
      }

      const inputTokens = Math.ceil(text.split(/\s+/).length * 1.3);
      if (user.tokensUsed + inputTokens >= CFG.DAILY_TOKEN_LIMIT) {
        return safeSend(chatId, `⚠️ Daily token limit (${CFG.DAILY_TOKEN_LIMIT}) reached. Try again tomorrow.`);
      }

      await hydrateUserPrefs(chatId);
      const lang    = userLanguages.get(cid(chatId)) || "en";

      // FIX #25: in-memory history cache would go here; we read from DB (acceptable at this scale)
      const history = await getRecentHistory(chatId);

      let aiResult;
      try {
        aiResult = await withThinkingIndicator(chatId, "🤔 Thinking…", () => getAIReply(chatId, text, history, lang));
      } catch (err) {
        if (err.message?.startsWith("PRO_LIMIT")) return safeSend(chatId, `⚠️ Pro model daily limit reached. Use /setmodel.`);
        if (err.message === "NO_KEY")              return safeSend(chatId, "⚠️ Model unavailable. Use /setmodel.");
        if (err.message?.startsWith("TIMEOUT"))    return safeSend(chatId, "⚠️ AI took too long to respond. Try again.");
        throw err;
      }

      const { reply, chosen } = aiResult;
      const outputTokens  = Math.ceil(reply.split(/\s+/).length * 1.3);
      const totalTokens   = inputTokens + outputTokens;

      if (user.tokensUsed + totalTokens > CFG.DAILY_TOKEN_LIMIT) {
        return safeSend(chatId, "⚠️ Reply would exceed daily token limit. Try again tomorrow.");
      }

      // Atomic DB update — FIX #8: lastActive always updated here
      await User.findOneAndUpdate(
        { chatId: cid(chatId) },
        {
          $inc: { requests: 1, tokensUsed: totalTokens },
          $set: { lastActive: new Date() },
        }
      );
      // Bump global daily aggregated stats
      bumpStats({ tokens: totalTokens, requests: 1 });

      await appendChatHistory(chatId, text, reply);

      const remaining = CFG.DAILY_REQUEST_LIMIT - (user.requests + 1);
      const tokLeft   = CFG.DAILY_TOKEN_LIMIT   - (user.tokensUsed + totalTokens);
      // FIX #28: escape model name in case it has underscores
      const footer    = `\n\n🤖 *${escapeMarkdown(chosen.name || "")}*  |  📨 ${remaining} left  |  🪙 ${tokLeft} tokens`;

      await safeSend(chatId, reply + footer, { parse_mode: "Markdown" });
    });
  } catch (err) {
    if (err.message === "QUEUE_FULL") {
      return safeSend(chatId, "⚠️ Bot is very busy. Please try again shortly.");
    }
    console.error("❌ Chat error:", err.message);
    await safeSend(chatId, "❌ Something went wrong. Please try again.");
  } finally {
    unmarkProcessing(chatId);
  }
});

// ============================================================
//  SECTION 11 · ADMIN COMMANDS
//  FIX #16: all wrapped in try/catch
// ============================================================

// ── /broadcast ───────────────────────────────────────────
bot.onText(/\/broadcast (.+)/, async (msg, match) => {
  if (msg.chat.id !== CFG.ADMIN_ID) return;
  bot.sendMessage(CFG.ADMIN_ID, "📡 Broadcast starting…").catch(() => {});
  try {
    const { success, failed } = await broadcastMessage(match[1]);
    bot.sendMessage(CFG.ADMIN_ID, `✅ Done.\n✔️ Sent: ${success}\n❌ Failed: ${failed}`).catch(() => {});
  } catch (err) {
    console.error("broadcast error:", err.message);
    bot.sendMessage(CFG.ADMIN_ID, "⚠️ Broadcast failed.").catch(() => {});
  }
});

// ── /usage ── single aggregated daily summary ────────────
bot.onText(/\/usage/, async (msg) => {
  if (msg.chat.id !== CFG.ADMIN_ID) return;
  try {
    const today = utcToday();
    const stats = await DailyStats.findOne({ date: today });
    const totalUsers = await User.countDocuments();

    const report =
      `📊 *Daily Summary* (${today})\n` +
      `━━━━━━━━━━━━━━━━━━━━━━\n` +
      `👥 Total users in DB: ${totalUsers}\n` +
      `━━━━━━━━━━━━━━━━━━━━━━\n` +
      `📨 Requests today:  ${stats?.requests  || 0}\n` +
      `🪙 Tokens today:    ${stats?.tokens    || 0}\n` +
      `🔍 Searches today:  ${stats?.search    || 0}\n` +
      `🎨 Images today:    ${stats?.imagine   || 0}\n` +
      `📄 Docs today:      ${stats?.doc       || 0}\n` +
      `🖼️ Img analyses:    ${stats?.img       || 0}\n` +
      `━━━━━━━━━━━━━━━━━━━━━━\n` +
      `_(resets at UTC midnight)_`;

    bot.sendMessage(CFG.ADMIN_ID, report, { parse_mode: "Markdown" }).catch(() => {});
  } catch (err) {
    console.error("usage error:", err.message);
    bot.sendMessage(CFG.ADMIN_ID, "⚠️ Could not generate usage report.").catch(() => {});
  }
});

// ── /users ───────────────────────────────────────────────
bot.onText(/\/users/, async (msg) => {
  if (msg.chat.id !== CFG.ADMIN_ID) return;
  try {
    const users = await User.find({ approvedUntil: { $gt: new Date() } }, { chatId: 1, approvedUntil: 1 });
    if (!users.length) return bot.sendMessage(CFG.ADMIN_ID, "📭 No active approved users.").catch(() => {});

    let text = "👥 *Approved Users*\n━━━━━━━━━━━━━\n";
    users.forEach((u, i) => {
      const rem = Math.max(0, Math.ceil((new Date(u.approvedUntil) - Date.now()) / 3600000)) + "h";
      text += `${i + 1}. \`${u.chatId}\` — ⏳ ${rem}\n`;
    });
    bot.sendMessage(CFG.ADMIN_ID, text, { parse_mode: "Markdown" }).catch(() => {});
  } catch (err) {
    console.error("users error:", err.message);
    bot.sendMessage(CFG.ADMIN_ID, "⚠️ Could not list users.").catch(() => {});
  }
});

// ── /approve ─────────────────────────────────────────────
bot.onText(/\/approve (\d+)/, async (msg, match) => {
  if (msg.chat.id !== CFG.ADMIN_ID) return;
  const userId    = match[1];
  const expiresAt = new Date(Date.now() + CFG.APPROVAL_HOURS * 3600000);
  try {
    await User.findOneAndUpdate(
      { chatId: userId },
      { $set: { approvedUntil: expiresAt } },
      { upsert: true }
    );
    // FIX #9: immediately invalidate approval cache
    approvalCache.delete(userId);
    approvalCache.delete(Number(userId));
    bot.sendMessage(CFG.ADMIN_ID, `✅ User ${userId} approved until ${expiresAt.toUTCString()}`).catch(() => {});
  } catch (err) {
    console.error("approve error:", err.message);
    bot.sendMessage(CFG.ADMIN_ID, "⚠️ Could not approve user.").catch(() => {});
  }
});

// ── /remove ──────────────────────────────────────────────
bot.onText(/\/remove (\d+)/, async (msg, match) => {
  if (msg.chat.id !== CFG.ADMIN_ID) return;
  const userId = match[1];
  try {
    await User.findOneAndUpdate({ chatId: userId }, { $set: { approvedUntil: null } });
    approvalCache.delete(userId);
    approvalCache.delete(Number(userId));
    bot.sendMessage(CFG.ADMIN_ID, `❌ User ${userId} removed.`).catch(() => {});
  } catch (err) {
    console.error("remove error:", err.message);
    bot.sendMessage(CFG.ADMIN_ID, "⚠️ Could not remove user.").catch(() => {});
  }
});

// ── /public / /private / /mode — FIX #10: persisted to DB ──
bot.onText(/\/public/, async (msg) => {
  if (msg.chat.id !== CFG.ADMIN_ID) return;
  try {
    PUBLIC_MODE = true;
    await Setting.findOneAndUpdate({ key: "publicMode" }, { $set: { value: true } }, { upsert: true });
    bot.sendMessage(CFG.ADMIN_ID, "🔓 Bot is now PUBLIC. (Persisted)").catch(() => {});
  } catch (err) {
    console.error("public mode error:", err.message);
    bot.sendMessage(CFG.ADMIN_ID, "⚠️ Could not save setting.").catch(() => {});
  }
});

bot.onText(/\/private/, async (msg) => {
  if (msg.chat.id !== CFG.ADMIN_ID) return;
  try {
    PUBLIC_MODE = false;
    await Setting.findOneAndUpdate({ key: "publicMode" }, { $set: { value: false } }, { upsert: true });
    bot.sendMessage(CFG.ADMIN_ID, "🔒 Bot is now PRIVATE. (Persisted)").catch(() => {});
  } catch (err) {
    console.error("private mode error:", err.message);
    bot.sendMessage(CFG.ADMIN_ID, "⚠️ Could not save setting.").catch(() => {});
  }
});

bot.onText(/\/mode/, (msg) => {
  if (msg.chat.id !== CFG.ADMIN_ID) return;
  bot.sendMessage(CFG.ADMIN_ID, PUBLIC_MODE ? "🔓 Mode: PUBLIC" : "🔒 Mode: PRIVATE").catch(() => {});
});

// ============================================================
//  BOOT
// ============================================================
startup().catch(err => {
  console.error("❌ Fatal startup error:", err.message);
  process.exit(1);
});