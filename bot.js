// telegram-bot.js
import express from 'express';
import TelegramBot from "node-telegram-bot-api";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";
import mongoose from 'mongoose';
import fetch from "node-fetch";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

import { createRequire } from "module";
const require = createRequire(import.meta.url);

const pdf = require("pdf-parse");
const mammoth = require("mammoth");

dotenv.config();

// --- CONFIGURATION FROM ENV ---
const DAILY_REQUEST_LIMIT = parseInt(process.env.DAILY_REQUEST_LIMIT);
const DAILY_TOKEN_LIMIT = parseInt(process.env.DAILY_TOKEN_LIMIT);
const MAX_REPLY_TOKENS = parseInt(process.env.MAX_REPLY_TOKENS);

const HISTORY_MESSAGES = parseInt(process.env.HISTORY_MESSAGES);
const DB_MSG_LIMIT = parseInt(process.env.DB_MSG_LIMIT);

const RATE_LIMIT = (parseInt(process.env.RATE_LIMIT_MS)) * 1000;
const COMMAND_LIMIT = (parseInt(process.env.COMMAND_LIMIT_MS)) * 1000;

const SEARCH_LIMIT = parseInt(process.env.SEARCH_LIMIT);
const SEARCH_RESULTS = parseInt(process.env.SEARCH_RESULTS) || 5;
const IMAGINE_LIMIT = parseInt(process.env.IMAGINE_LIMIT);
const DOC_LIMIT = parseInt(process.env.LIMIT_DOC_ANALYSIS);
const LIMIT_IMG = parseInt(process.env.LIMIT_IMG_ANALYSIS);
const LIMIT_PRO = parseInt(process.env.LIMIT_PRO_MODEL);

const APPROVAL_HOURS = parseFloat(process.env.APPROVAL_EXPIRY_HOURS);

// 🔒 GLOBAL SAFETY NETS
process.on("unhandledRejection", (err) => {
  console.error("Unhandled rejection:", err);
});

process.on("uncaughtException", (err) => {
  console.error("Uncaught exception:", err);
});

const PORT = process.env.PORT || 3000;

// 🔥 make bot public(true) or private(false) 
let PUBLIC_MODE = false;


const ADMIN_ID = Number(process.env.ADMIN_ID);
const FORCE_JOIN_CHANNEL = process.env.FORCE_JOIN_CHANNEL || "";

//membership check function
async function isUserMember(chatId) {
  try {
    const member = await bot.getChatMember(FORCE_JOIN_CHANNEL, chatId);
    const status = member.status;
    // "creator", "administrator", "member" are valid
    return ["creator", "administrator", "member"].includes(status);
  } catch (err) {
    console.warn("⚠️ Membership check failed:", err.message);
    return false; // treat errors as not a member
  }
}

// ✅ New Helper Function
async function checkMembership(msg) {
  const chatId = msg.chat.id;
  if (await isUserMember(chatId)) return true; // User is member, allow

  // User is NOT member, send warning
  bot.sendMessage(chatId, `⚠️ You must join our channel first to use this bot.`, {
    reply_markup: {
      inline_keyboard: [
        [{ text: "📢 Join Channel", url: `https://t.me/${FORCE_JOIN_CHANNEL.replace("@", "")}` }]
      ]
    }
  });
  return false; // Block execution
}

// 🔒 Check approval from Database
async function isUserApproved(chatId) {
  // 1. Always allow Admin and Public Mode
  if (PUBLIC_MODE) return true;
  if (chatId === ADMIN_ID) return true;

  // 2. Check Database
  const user = await User.findOne({ chatId });

  // 3. If no user or no date set -> Not Approved
  if (!user || !user.approvedUntil) return false;

  // 4. If date is in the past -> Expired
  if (new Date() > user.approvedUntil) return false;

  return true; // ✅ Approved
}


// chatId -> last request timestamp (ms)
const rateLimitMap = new Map();
const RATE_LIMIT_MS = RATE_LIMIT

//limit on message 
function guardRateLimit(msg) {
  // ❌ Skip commands
  if (msg.text && msg.text.startsWith("/")) {
    return true;
  }

  // ❌ Skip media here (handled elsewhere)
  if (msg.photo || msg.document) {
    return true;
  }

  const chatId = msg.chat.id;
  const now = Date.now();
  const last = rateLimitMap.get(chatId) || 0;

  const remaining = RATE_LIMIT_MS - (now - last);

  if (remaining > 0) {
    const seconds = Math.ceil(remaining / 1000);
    bot.sendMessage(chatId, `⏳ Please wait ${seconds}s before sending another request.`);
    return false;
  }

  rateLimitMap.set(chatId, now);
  return true;
}

//limit on media 
const mediaRateLimit = new Map();
// NEW: Media-specific rate limit function
function guardRateLimitMedia(msg) {
  const chatId = msg.chat.id;
  const now = Date.now();

  // 1. Get the limit from .env
  const COOLDOWN = (parseInt(process.env.LIMIT_MEDIA_COOLDOWN)) * 1000;

  // 2. Check the MEDIA map, not the text map
  const last = mediaRateLimit.get(chatId) || 0;
  const remaining = COOLDOWN - (now - last);

  if (remaining > 0) {
    const seconds = Math.ceil(remaining / 1000);
    bot.sendMessage(chatId, `⏳ Please wait ${seconds}s before sending another file.`);
    return false;
  }

  // 3. Update the MEDIA map
  mediaRateLimit.set(chatId, now);
  return true;
}

//limit on search and imagine
const commandRateLimit = new Map();
const COMMAND_LIMIT_MS = COMMAND_LIMIT
function guardCommandRateLimit(msg, commandName) {
  const chatId = msg.chat.id;
  const key = `${chatId}:${commandName}`;
  const now = Date.now();
  const last = commandRateLimit.get(key) || 0;

  if (now - last < COMMAND_LIMIT_MS) {
    const wait = Math.ceil((COMMAND_LIMIT_MS - (now - last)) / 1000);
    bot.sendMessage(chatId, `⏳ Please wait ${wait}s before using /${commandName} again.`);
    return false;
  }

  commandRateLimit.set(key, now);
  return true;
}


async function downloadFile(fileId) {
  const file = await bot.getFile(fileId);
  const fileUrl = `https://api.telegram.org/file/bot${process.env.TELEGRAM_TOKEN}/${file.file_path}`;
  const res = await fetch(fileUrl);
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  return buf;
}

const MODELS = {
  gemini: {
    name: process.env.GEMINI_MODEL_1,
    key: process.env.GEMINI_API_KEY,
    provider: "gemini",
    model: process.env.GEMINI_MODEL_1,
  },
  gemini_flash1: {
    name: process.env.GEMINI_MODEL_3,
    key: process.env.GEMINI_API_KEY,
    provider: "gemini",
    model: process.env.GEMINI_MODEL_3,
  },
  gemini_flash2: {
    name: process.env.GEMINI_MODEL_2,
    key: process.env.GEMINI_API_KEY,
    provider: "gemini",
    model: process.env.GEMINI_MODEL_2,
  },
  gemini_flash3: {
    name: process.env.GEMINI_MODEL_5,
    key: process.env.GEMINI_API_KEY,
    provider: "gemini",
    model: process.env.GEMINI_MODEL_5,
  },
  gemini_pro: {
    name: process.env.GEMINI_MODEL_4,
    key: process.env.GEMINI_API_KEY,
    provider: "gemini",
    model: process.env.GEMINI_MODEL_4,
  },
  openai: {
    name: process.env.OPENAI_MODEL_1,
    key: process.env.OPENAI_API_KEY,
    type: "openai"
  },
  claude: {
    name: process.env.CLAUDE_MODEL_1,
    key: process.env.CLAUDE_API_KEY,
    type: "claude"
  },
};

const userSelectedModel = {}; // { chatId: "gemini" }


mongoose.connect(process.env.MONGODB_CONNECT, { dbName: "Telegram" }).then((req, res) => {
  console.log("MongoDb is Connected....");
});

const userSchema = new mongoose.Schema({
  chatId: { type: String, required: true, unique: true },
  approvedUntil: { type: Date, default: null },
  messages: [
    {
      role: { type: String, enum: ["user", "bot"], required: true },
      text: String,
      timestamp: { type: Date, default: Date.now }
    }
  ],
  requests: { type: Number, default: 0 },        // daily requests
  lastReset: { type: Date, default: Date.now },  // daily reset for requests
  usage: {
    tokensUsed: { type: Number, default: 0 },   // total tokens used today
    resetDate: { type: Date, default: Date.now } // when to reset quota
  }
});

const User = mongoose.model("User", userSchema);



const app = express();

// -- set view engine to ejs --
app.set("view engine", "ejs");

// -- ejs files
app.get("/", (req, res) => res.render("home"));
app.get("/privacy", (req, res) => res.render("privacy"));
app.get("/terms", (req, res) => res.render("terms"));


// --- Load API keys from .env ---
const TELEGRAM_TOKEN = process.env.TELEGRAM_TOKEN;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const RENDER_URL = process.env.RENDER_URL;


// --- Initialize Telegram Bot ---
const bot = new TelegramBot(TELEGRAM_TOKEN);

// 2️⃣ Set the webhook URL
bot.setWebHook(`${RENDER_URL}/bot${TELEGRAM_TOKEN}`);

// 3️⃣ Express route to receive updates
app.use(express.json());
app.post(`/bot${TELEGRAM_TOKEN}`, (req, res) => {
  bot.processUpdate(req.body);
  res.sendStatus(200);
});

// --- Running on this port ---
app.listen(PORT, () => console.log(`✅ Web server running on port ${PORT}`));

//--- reply keywords ---
const mainKeyboard = {
  reply_markup: {
    keyboard: [
      ["🔍 Search", "🎨 Imagine"],
      ["🚫 Report Error", "📄 Document Analysis"],
    ],
    resize_keyboard: true,
    one_time_keyboard: false
  }
};


// --- Register Commands with Telegram ---
bot.setMyCommands([
  { command: "start", description: "🚀Check if bot is avtive" },
  { command: "help", description: "📝List of commands" },
  { command: "account", description: "👤 My account info" },
  { command: "language", description: "🌐 Change language" },
  { command: "setmodel", description: "🤖 Select Ai Models" },
  { command: "clearchat", description: "🧹 Clear chat history" },
  { command: "about", description: "👀About this bot" },
  { command: "terms", description: "📜 Terms of service" },
  { command: "status", description: "✅ Check your bot access" },

]);


// --- Initialize Gemini ---
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

// Define supported languages
const LANGUAGES = {
  en: "🇬🇧 English",
  hi: "🇮🇳 Hindi",
  es: "🇪🇸 Spanish",
  fr: "🇫🇷 French",
  de: "🇩🇪 German",
  ja: "🇯🇵 Japanese",
  ru: "🇷🇺 Russian",
  ar: "🇸🇦 Arabic",
};


//usage limit 
const userUsage = {};

function getToday() {
  return new Date().toISOString().slice(0, 10);
}

function checkLimit(chatId, type, limit) {
  const today = getToday();
  if (!userUsage[chatId] || userUsage[chatId].date !== today) {
    userUsage[chatId] = { date: today, search: 0, imagine: 0, doc: 0, img: 0, proTokens: 0 };
  }

  if (userUsage[chatId][type] >= limit) return false;
  userUsage[chatId][type]++;
  return true;
}

// --- Commands ---
bot.onText(/\/start/, async (msg) => {
  if (!await checkMembership(msg)) return;

  bot.sendMessage(msg.chat.id, `👋 Hi ${msg.from.first_name}!
I am your Private AI assistant.
Type any question and I'll try to answer`, mainKeyboard
  );
});


bot.onText(/\/help/, async (msg) => {
  // 1. Check if user is a member of the channel first
  if (!await checkMembership(msg)) return;

  const chatId = msg.chat.id;

  // 2. Define the Standard Help Message (Visible to everyone)
  let helpMessage = `📌 *Available commands:*
/start - Start the bot
/status - Check access status
/help - Show this help menu
/about - About this bot
/clearchat - Clear chat history
/terms - Terms of service
/account - My account info
/setmodel - Choose AI Model
/language - Change language
/imagine A dancing panda 
/search Today's latest news 

💡 *Tip:* Send any document or photo for analysis!
`;

  // 3. Append Admin Commands ONLY if the user is the Admin
  if (chatId === ADMIN_ID) {
    helpMessage += `
🎧 *Admin Commands:*
/broadcast - Send message to all users
/usage - Check usage report
/approve - Approve a user manually
/remove - Remove user approval
/users - List approved users
/mode - Check Private/Public mode
/private - Set bot to Private mode
/public - Set bot to Public mode
`;
  }

  // 4. Send the message
  bot.sendMessage(chatId, helpMessage, { parse_mode: "Markdown" });
});


bot.onText(/\/about/, async (msg) => {
  if (!await checkMembership(msg)) return;

  bot.sendMessage(msg.chat.id, `🤖 This bot is built with:
- Telegram Bot API
- Google Gemini API
- OpenAi API
- Claude API
- MongoDB for data storage
- Node.js`);
});


bot.onText(/\/clearchat/, async (msg) => {
  const chatId = msg.chat.id;

  if (!await checkMembership(msg)) return;

  let user = await User.findOne({ chatId });
  if (!user) {
    bot.sendMessage(chatId, "⚠️ No chat history found.");
    return;
  }

  // ✅ Clear all saved messages
  user.messages = [];
  await user.save();

  bot.sendMessage(chatId, "🧹 Your chat history has been cleared.");
});


bot.onText(/\/terms/, async (msg) => {
  const chatId = msg.chat.id;

  if (!await checkMembership(msg)) return;

  const terms = `
  📜 *Terms of Service*
  
  1. This bot is provided for educational and personal use only.  
  2. Do not use this bot to share harmful, illegal, or inappropriate content.  
  3. The bot may store limited usage data to improve responses and enforce usage limits.  
  4. Responses are generated by AI (Gemini API) and may not always be accurate.  
  5. By using this bot, you agree to these terms.
  `;

  bot.sendMessage(chatId, terms, {
    parse_mode: "Markdown",
    reply_markup: {
      inline_keyboard: [
        [
          { text: "📄 Full Terms of Service", url: `${RENDER_URL}/terms` },
          { text: "🔒 Privacy Policy", url: `${RENDER_URL}/privacy` }
        ]
      ]
    }
  });

});


bot.onText(/\/approve (\d+)/, async (msg, match) => {
  if (msg.chat.id !== ADMIN_ID) return;

  const userId = match[1];
  const expiresAt = new Date(Date.now() + (APPROVAL_HOURS * 60 * 60 * 1000));

  // Update MongoDB (Create user if missing, set new date)
  await User.findOneAndUpdate(
    { chatId: userId },
    { approvedUntil: expiresAt },
    { upsert: true, new: true }
  );

  bot.sendMessage(ADMIN_ID, `✅ User ${userId} approved until ${expiresAt.toLocaleString()}`);
});


async function guardAccess(msg) {
  const chatId = msg.chat.id;
  const approved = await isUserApproved(chatId); // 👈 Checks DB now

  if (!approved) {
    bot.sendMessage(chatId, `🛡️ *Member Only*\nSorry, this feature is currently locked. Please reach out to ${FORCE_JOIN_CHANNEL} to get your account approved!`);
    return false;
  }
  return true;
}

// --- My Account ---
bot.onText(/\/account/, async (msg) => {
  const chatId = msg.chat.id;

  if (!await checkMembership(msg)) return;

  // Fetch user from DB
  let user = await User.findOne({ chatId });
  if (!user) {
    user = new User({
      chatId,
      requests: 0,
      lastReset: new Date(),
      usage: { tokensUsed: 0, resetDate: new Date() },
      messages: [],
    });
    await user.save();
  }

  // ✅ check todays date
  const todayDate = new Date().toDateString();

  // 1. Reset Requests if day changed
  if (user.lastReset.toDateString() !== todayDate) {
    user.requests = 0;
    user.lastReset = new Date();
  }

  // 2. Reset Tokens if day changed
  if (user.usage.resetDate.toDateString() !== todayDate) {
    user.usage.tokensUsed = 0;
    user.usage.resetDate = new Date();
  }

  // 3. Save the reset immediately so the numbers are correct
  await user.save();

  const remainingRequests = DAILY_REQUEST_LIMIT - user.requests;
  const usedTokens = user.usage?.tokensUsed || 0;
  const remainingTokens = DAILY_TOKEN_LIMIT - usedTokens;
  const lang = userLanguages[chatId] || "en";

  // Calculate midnight reset time
  const resetTime = new Date();
  resetTime.setHours(24, 0, 0, 0); // midnight

  // ---- NEW PART: usage limits tracking ----
  const today = getToday();
  const usage = userUsage[chatId] || { date: today, search: 0, imagine: 0, doc: 0, img: 0, proTokens: 0 };
  if (usage.date !== today) {
    usage.search = 0;
    usage.imagine = 0;
    usage.doc = 0;
    usage.img = 0;
    usage.proTokens = 0;
    usage.date = today;
  }

  const limits = {
    search: SEARCH_LIMIT,
    imagine: IMAGINE_LIMIT,
    doc: DOC_LIMIT,
    img: LIMIT_IMG,
    proTokens: LIMIT_PRO
  };

  bot.sendMessage(
    chatId,
    `
👤 *My Account*
━━━━━━━━━━━━━
- Requests used today: ${user.requests}
- Requests remaining: ${remainingRequests}
- Daily request limit: ${DAILY_REQUEST_LIMIT}

- Tokens used today: ${usedTokens}
- Tokens remaining: ${remainingTokens}
- Daily token limit: ${DAILY_TOKEN_LIMIT}
- Max tokens per reply: ${MAX_REPLY_TOKENS}

🌍 Current language: ${LANGUAGES[lang]} (${lang})

🕒 Quota resets at: ${resetTime.toLocaleString()}

📊 *Feature Usage Today*
━━━━━━━━━━━━━
🔍 Searches: ${usage.search}/${limits.search}
🎨 Image Generations: ${usage.imagine}/${limits.imagine}
📄 Document Analyses: ${usage.doc}/${limits.doc}
🖼️ Image Analyses: ${usage.img}/${limits.img}
🤖 Pro-Model Requests: ${usage.proTokens}/${limits.proTokens}
    `,
    { parse_mode: "Markdown" }
  );
});


bot.onText(/\/language/, async (msg) => {
  const chatId = msg.chat.id;

  if (!await checkMembership(msg)) return;

  // Convert LANGUAGES object to inline keyboard (2 buttons per row)
  const buttons = Object.entries(LANGUAGES).map(([code, name]) => {
    return { text: name, callback_data: `lang_${code}` };
  });

  // Split into rows of 2 buttons
  const keyboard = [];
  for (let i = 0; i < buttons.length; i += 2) {
    keyboard.push(buttons.slice(i, i + 2));
  }

  bot.sendMessage(chatId, "🌐 Choose your language:", {
    reply_markup: { inline_keyboard: keyboard }
  });
});

const userLanguages = {}; // store user language in-memory

bot.on("callback_query", (query) => {
  const chatId = query.message.chat.id;

  if (query.data.startsWith("lang_")) {
    const lang = query.data.replace("lang_", "");
    userLanguages[chatId] = lang;

    bot.sendMessage(chatId, `✅ Language changed to: ${LANGUAGES[lang]}`);
    bot.answerCallbackQuery(query.id);
  }
});

//search the web
bot.onText(/\/search (.+)/, async (msg, match) => {
  const chatId = msg.chat.id;

  if (!await checkMembership(msg)) return;

  if (!await guardAccess(msg)) return;

  if (!guardCommandRateLimit(msg, "search")) return;

  const query = match[1];
  bot.sendChatAction(chatId, "typing");

  //added Search limit 
  if (!checkLimit(chatId, "search", SEARCH_LIMIT)) {
    bot.sendMessage(chatId, `⚠️ Daily search limit reached (${SEARCH_LIMIT}). Try again tomorrow.`);
    return;
  }


  try {
    const res = await fetch("https://google.serper.dev/search", {
      method: "POST",
      headers: {
        "X-API-KEY": process.env.SERPER_API_KEY,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ q: query })
    });

    if (!res.ok) {
      throw new Error(`Serper API error: ${res.status}`);
    }

    const data = await res.json();

    const results = data.organic?.slice(0, SEARCH_RESULTS)
      .map((r, i) => `${i + 1}. [${r.title}](${r.link})\n${r.snippet}`)
      .join("\n\n");

    if (!results) {
      bot.sendMessage(chatId, "⚠ No search results found.");
      return;
    }

    bot.sendMessage(chatId, `🔍 *Search results for:* ${query}\n\n${results}`, {
      parse_mode: "Markdown",
      disable_web_page_preview: true
    });

  } catch (err) {
    console.error("❌ Search error:", err);
    bot.sendMessage(chatId, "⚠ Could not perform web search.");
  }
});


//select model
bot.onText(/\/setmodel/, async (msg) => {
  const chatId = msg.chat.id;

  if (!await checkMembership(msg)) return;

  // Build buttons dynamically only for models with keys
  const buttons = Object.entries(MODELS)
    .map(([id, m]) => ({
      text: m.name + (m.key ? "" : " ❌ Unavailable"),
      callback_data: m.key ? `model_${id}` : `unavailable_${id}`
    }))
    .map(b => [b]); // one button per row

  bot.sendMessage(chatId, "🤖 Choose your AI model:", {
    reply_markup: { inline_keyboard: buttons }
  });
});


bot.on("callback_query", async (query) => {
  const chatId = query.message.chat.id;

  if (query.data.startsWith("model_")) {
    const modelId = query.data.replace("model_", "");
    userSelectedModel[chatId] = modelId;

    bot.sendMessage(chatId, `✅ Model set to *${MODELS[modelId].name}*`, { parse_mode: "Markdown" });
    bot.answerCallbackQuery(query.id);
  }

  if (query.data.startsWith("unavailable_")) {
    bot.answerCallbackQuery(query.id, {
      text: "⚠ This model is not available right now.",
      show_alert: true
    });
  }
});


//imagine 
bot.onText(/\/imagine (.+)/, async (msg, match) => {
  const chatId = msg.chat.id;

  if (!await checkMembership(msg)) return;

  if (!await guardAccess(msg)) return;

  if (!guardCommandRateLimit(msg, "imagine")) return;

  const prompt = match[1];

  bot.sendChatAction(chatId, "upload_photo");

  //added imagine limit  
  if (!checkLimit(chatId, "imagine", IMAGINE_LIMIT)) {
    bot.sendMessage(chatId, `⚠️ Daily image generation limit reached (${IMAGINE_LIMIT}). Try again tomorrow.`);
    return;
  }


  try {
    const imageUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}`;

    await bot.sendPhoto(chatId, imageUrl, {
      caption: `🎨 Prompt: ${prompt}`
    });
  } catch (err) {
    console.error("❌ Image generation error:", err);
    bot.sendMessage(chatId, "⚠ Could not generate image.");
  }
});

//document analysis
bot.on("document", async (msg) => {
  const chatId = msg.chat.id;

  if (!await checkMembership(msg)) return;

  if (!await guardAccess(msg)) return;

  if (!guardRateLimitMedia(msg)) return;

  const fileName = msg.document.file_name.toLowerCase();

  bot.sendChatAction(chatId, "typing");

  //added document analysis limit
  if (!checkLimit(chatId, "doc", DOC_LIMIT)) {
    bot.sendMessage(chatId, `⚠️ Daily document analysis limit reached (${DOC_LIMIT}). Try again tomorrow.`);
    return;
  }


  try {
    const fileBuffer = await downloadFile(msg.document.file_id);
    if (!Buffer.isBuffer(fileBuffer) || fileBuffer.length === 0) {
      bot.sendMessage(chatId, "⚠️ Could not download your document.");
      return;
    }

    let text = "";
    if (fileName.endsWith(".pdf")) {
      const data = await pdf(fileBuffer);
      text = data.text;
    } else if (fileName.endsWith(".docx")) {
      const data = await mammoth.extractRawText({ buffer: fileBuffer });
      text = data.value;
    } else if (fileName.endsWith(".txt")) {
      text = fileBuffer.toString("utf-8");
    } else {
      bot.sendMessage(chatId, "⚠️ Only PDF, DOCX or TXT files are supported.");
      return;
    }

    if (!text.trim()) {
      bot.sendMessage(chatId, "⚠️ No readable text found in this file.");
      return;
    }

    const DOC_CHAR_LIMIT = parseInt(process.env.DOC_CHAR_LIMIT);
    if (text.length > DOC_CHAR_LIMIT) text = text.slice(0, DOC_CHAR_LIMIT);

    const docModel = genAI.getGenerativeModel({ model: process.env.GEMINI_MODEL_5 });
    const result = await docModel.generateContent({
      contents: [
        { role: "user", parts: [{ text: `Summarize this document:\n\n${text}` }] }
      ],
      generationConfig: { maxOutputTokens: MAX_REPLY_TOKENS }
    });

    const reply = result?.response?.text() || "⚠️ No response from Gemini.";
    bot.sendMessage(chatId, "📄 Document summary:\n\n" + reply);

  } catch (err) {
    console.error("❌ Document analysis error:", err);
    bot.sendMessage(chatId, "⚠️ Could not analyze your document.");
  }
});

//image analysis
bot.on("photo", async (msg) => {
  const chatId = msg.chat.id;

  if (!await checkMembership(msg)) return;

  if (!await guardAccess(msg)) return;

  if (!guardRateLimitMedia(msg)) return;

  const photo = msg.photo[msg.photo.length - 1]; // largest size

  bot.sendChatAction(chatId, "typing");

  //added image analysis limit
  if (!checkLimit(chatId, "img", LIMIT_IMG)) {
    bot.sendMessage(chatId, `⚠️ Daily image analysis limit reached (${LIMIT_IMG}). Try again tomorrow.`);
    return;
  }


  try {
    const fileBuffer = await downloadFile(photo.file_id);
    const base64Image = fileBuffer.toString("base64");

    const imgModel = genAI.getGenerativeModel({ model: process.env.GEMINI_MODEL_5 });
    const result = await imgModel.generateContent([
      "Describe this image clearly.",
      { inlineData: { data: base64Image, mimeType: "image/jpeg" } }
    ]);

    const reply = result?.response?.text() || "⚠️ No response from Gemini.";
    bot.sendMessage(chatId, "🖼️ Image analysis:\n\n" + reply);

  } catch (err) {
    console.error("❌ Image analysis error:", err);
    bot.sendMessage(chatId, "⚠️ Could not analyze your image.");
  }
});

// Function to split long messages into chunks
async function sendLongMessage(chatId, text) {
  const MAX_LENGTH = 4000;

  for (let i = 0; i < text.length; i += MAX_LENGTH) {
    const chunk = text.substring(i, i + MAX_LENGTH);
    await bot.sendMessage(chatId, chunk);
  }
}

// --- Chat with Gemini ---
bot.on("message", async (msg) => {
  const chatId = msg.chat.id;
  const text = msg.text;

  if (text && text.startsWith("/")) return; // ignore commands & empty

  if (!text) return;

  if (!await checkMembership(msg)) return;

  if (!await guardAccess(msg)) return;

  if (!guardRateLimit(msg)) return;

  // Block pure link messages
  if (msg.text && /^https?:\/\//i.test(msg.text.trim())) {
    bot.sendMessage(chatId, "⚠️ Links are not allowed. Please send text, document, or image.");
    return;
  }


  //reply keywords
  if (text === "🔍 Search") {
    bot.sendMessage(chatId, "🔍 Please type your search like:\n`/search your query`", { parse_mode: "Markdown" });
    return;
  }

  if (text === "🎨 Imagine") {
    bot.sendMessage(chatId, "🎨 Please type your prompt like:\n`/imagine your prompt`", { parse_mode: "Markdown" });
    return;
  }

  if (text === "🚫 Report Error") {
    bot.sendMessage(chatId, `Contact ${FORCE_JOIN_CHANNEL} to report any problem.`);
    return;
  }

  if (text === "📄 Document Analysis") {
    bot.sendMessage(chatId, "Send any document and image for it's analysis.");
    return;
  }


  // Fetch or create user from DB
  let user = await User.findOne({ chatId });
  if (!user) {
    user = new User({
      chatId,
      requests: 0,
      lastReset: new Date(),
      usage: { tokensUsed: 0, resetDate: new Date() },
      messages: [],
    });
    await user.save();
  }

  // Reset request limits daily
  const today = new Date().toDateString();
  if (user.lastReset.toDateString() !== today) {
    user.requests = 0;
    user.lastReset = new Date();
  }

  // Reset token usage daily
  if (user.usage.resetDate.toDateString() !== today) {
    user.usage.tokensUsed = 0;
    user.usage.resetDate = new Date();
  }

  // Check daily request limit 
  if (user.requests >= DAILY_REQUEST_LIMIT) {
    bot.sendMessage(chatId, `⚠️ You’ve reached your daily request limit (${DAILY_REQUEST_LIMIT}). Try again tomorrow.`);
    return;
  }

  // Estimate input tokens
  if (typeof text !== "string") return;
  const inputTokens = Math.ceil(text.split(/\s+/).length * 1.3);

  // Check token limit per day
  if (user.usage.tokensUsed + inputTokens >= DAILY_TOKEN_LIMIT) {
    bot.sendMessage(chatId, `⚠️ You’ve reached your daily token limit (${DAILY_TOKEN_LIMIT}). Try again tomorrow.`);
    return;
  }

  // Language
  const lang = userLanguages[chatId] || "en";

  try {
    bot.sendChatAction(chatId, "typing");

    // Get last messages for context
    const history = user.messages
      .slice(-HISTORY_MESSAGES)
      .map((m) => `${m.role}: ${m.text}`)
      .join("\n");

    // Ask Gemini with output limit
    const selectedId = userSelectedModel[chatId] || "gemini";
    const chosen = MODELS[selectedId];

    // If model has no API key, block the request
    if (!chosen.key) {
      bot.sendMessage(chatId, "⚠ This model is not available right now. Please choose another using /setmodel");
      return;
    }

    let reply = "";

    if (chosen.provider === "gemini") {

      // Optional: keep Pro limit
      if (chosen.model === process.env.GEMINI_MODEL_4) {
        if (!checkLimit(chatId, "proTokens", LIMIT_PRO)) {
          bot.sendMessage(chatId, `⚠️ Pro model daily limit reached (${LIMIT_PRO})`);
          return;
        }
      }

      //console.log("ACTIVE MODEL:", chosen.model);

      const dynamicModel = genAI.getGenerativeModel({
        model: chosen.model
      });

      const result = await dynamicModel.generateContent({
        contents: [
          {
            role: "user",
            parts: [{ text: `Answer in ${LANGUAGES[lang]} (${lang})\n\nConversation so far:\n${history}\n\nUser: ${text}` }],
          },
        ],
        generationConfig: { maxOutputTokens: MAX_REPLY_TOKENS },
      });

      reply = result?.response?.text() || "⚠️ No response from Gemini.";
    } else if (chosen.type === "openai") {
      //if api key is not available then 
      if (!process.env.OPENAI_API_KEY) {
        bot.sendMessage(chatId, "⚠ This model is not available right now.");
        return;
      }
      //other wise normal response 
      const OpenAI = (await import("openai")).default;
      const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

      const result = await openai.chat.completions.create({
        model: process.env.OPENAI_MODEL_1,
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: `Answer in ${LANGUAGES[lang]} (${lang})\n\nConversation so far:\n${history}\n\nUser: ${text}` }
        ],
        max_tokens: MAX_REPLY_TOKENS
      });

      reply = result.choices[0].message.content || "⚠ No response from OpenAI.";
    } else if (chosen.type === "claude") {
      //if api is not available
      if (!process.env.CLAUDE_API_KEY) {
        bot.sendMessage(chatId, "⚠ This model is not available right now.");
        return;
      }
      //otherwise normal response
      const Anthropic = (await import("@anthropic-ai/sdk")).default;
      const anthropic = new Anthropic({ apiKey: process.env.CLAUDE_API_KEY });
      const response = await anthropic.messages.create({
        model: process.env.CLAUDE_MODEL_1,
        max_tokens: MAX_REPLY_TOKENS,
        messages: [
          {
            role: "user",
            content: `Answer in ${LANGUAGES[lang]} (${lang})\n\nConversation so far:\n${history}\n\nUser: ${text}`
          }
        ]
      });

      reply = response.content?.[0]?.text || "⚠ No response from Claude.";
    }


    // Estimate output tokens
    const outputTokens = Math.ceil(reply.split(/\s+/).length * 1.3);

    // Final check for token limit
    if (user.usage.tokensUsed + inputTokens + outputTokens > DAILY_TOKEN_LIMIT) {
      bot.sendMessage(chatId, `⚠️ This reply would exceed your daily token limit (${DAILY_TOKEN_LIMIT}). Try again tomorrow.`);
      return;
    }

    // Save conversation in MongoDB
    user.messages.push({ role: "user", text });
    user.messages.push({ role: "bot", text: reply });

    //limit number of chat saved
    if (user.messages.length > DB_MSG_LIMIT) {
      user.messages = user.messages.slice(-DB_MSG_LIMIT);
    }

    // Update usage
    user.requests += 1;
    user.usage.tokensUsed += inputTokens + outputTokens;
    await user.save();

    // Reply with usage info
    // NEW CODE - Uses the split function
    const footer = `\n\n🤖 Model: ${chosen.name}\n🪙 Requests left: ${DAILY_REQUEST_LIMIT - user.requests}\nTokens left: ${DAILY_TOKEN_LIMIT - user.usage.tokensUsed}`;
    const fullResponse = reply + footer;

    if (fullResponse.length > 4000) {
      // If too long, split it
      await sendLongMessage(chatId, fullResponse);
    } else {
      // Short message: Send as Markdown (Looks professional)
      // We use try/catch in case Gemini generates broken Markdown
      try {
        await bot.sendMessage(chatId, fullResponse, { parse_mode: "Markdown" });
      } catch (err) {
        // Fallback to plain text if Markdown fails
        await bot.sendMessage(chatId, fullResponse);
      }
    }
  } catch (err) {
    console.error("❌ Gemini error:", err);
    bot.sendMessage(chatId, "❌ BOT ERROR: " + (err.message || "Unknown error"));
  }
});

//broadcast 
bot.onText(/\/broadcast (.+)/, async (msg, match) => {
  if (msg.chat.id !== ADMIN_ID) return;

  const message = match[1];

  try {
    const users = await User.find({});
    let success = 0, failed = 0;

    for (const u of users) {
      try {
        await bot.sendMessage(u.chatId, `📢 Broadcast from Admin:\n\n${message}`);
        success++;
      } catch (err) {
        console.warn(`⚠️ Could not send to ${u.chatId}: ${err.message}`);
        failed++;
      }
    }

    bot.sendMessage(
      ADMIN_ID,
      `✅ Broadcast sent to ${success} users.\n⚠️ Failed to send to ${failed} users.`
    );
  } catch (err) {
    console.error("❌ Broadcast error:", err);
    bot.sendMessage(ADMIN_ID, "⚠️ Failed to send broadcast.");
  }
});

//usage
bot.onText(/\/usage/, async (msg) => {
  if (msg.chat.id !== ADMIN_ID) return;

  try {
    const users = await User.find({});
    const today = getToday();

    let report = `📊 *Usage Report* (${today})\n━━━━━━━━━━━━━\n`;

    for (const u of users) {
      const usage = userUsage[u.chatId] || { search: 0, imagine: 0, doc: 0, img: 0, proTokens: 0, date: today };

      report += `
👤 ID: \`${u.chatId}\`
Requests: ${u.requests}, Tokens: ${u.usage?.tokensUsed || 0}
🔍 ${usage.search || 0} | 🎨 ${usage.imagine || 0} | 📄 ${usage.doc || 0} | 🖼️ ${usage.img || 0} | 🤖 ${usage.proTokens || 0}
━━━━━━━━━━━━━`;
    }

    bot.sendMessage(ADMIN_ID, report, { parse_mode: "Markdown" });
  } catch (err) {
    console.error("❌ Usage report error:", err);
    bot.sendMessage(ADMIN_ID, "⚠️ Failed to generate usage report.");
  }
});

//see users
bot.onText(/\/users/, async (msg) => { // 👈 Mark as async
  if (msg.chat.id !== ADMIN_ID) return;

  // Find all users where approvedUntil is in the FUTURE ($gt = Greater Than)
  const users = await User.find({ approvedUntil: { $gt: new Date() } });

  if (users.length === 0) {
    return bot.sendMessage(ADMIN_ID, "📭 No active users.");
  }

  let text = "👥 *Approved Users*\n━━━━━━━━━━━━━\n";
  users.forEach((u, i) => {
    const remaining = Math.max(0, Math.ceil((new Date(u.approvedUntil) - Date.now()) / (3600000))) + "h";
    text += `${i + 1}. \`${u.chatId}\` — ⏳ ${remaining}\n`;
  });

  bot.sendMessage(ADMIN_ID, text, { parse_mode: "Markdown" });
});

//remove users
bot.onText(/\/remove (\d+)/, async (msg, match) => {
  if (msg.chat.id !== ADMIN_ID) return;
  const userId = match[1];

  // Set date to null to remove access
  await User.findOneAndUpdate({ chatId: userId }, { approvedUntil: null });

  bot.sendMessage(ADMIN_ID, `❌ User ${userId} removed.`);
});

//user status
bot.onText(/\/status/, async (msg) => {
  const chatId = msg.chat.id;

  if (PUBLIC_MODE) {
    bot.sendMessage(chatId, "🔓 Bot is currently in PUBLIC MODE");
  }

  if (await isUserApproved(chatId)) {
    bot.sendMessage(chatId, "✅ You have access to this bot.");
  } else {
    bot.sendMessage(chatId, "🚧 You do not have access.");
  }
});

//set bot public
bot.onText(/\/public/, (msg) => {
  if (msg.chat.id !== ADMIN_ID) return;

  PUBLIC_MODE = true;
  bot.sendMessage(msg.chat.id, "🔓 Bot is now in PUBLIC mode");
});

//set bot private 
bot.onText(/\/private/, (msg) => {
  if (msg.chat.id !== ADMIN_ID) return;

  PUBLIC_MODE = false;
  bot.sendMessage(msg.chat.id, "🔒 Bot is now in PRIVATE mode");
});

//set mode private or public
bot.onText(/\/mode/, (msg) => {
  if (msg.chat.id !== ADMIN_ID) return;

  bot.sendMessage(
    msg.chat.id,
    PUBLIC_MODE
      ? "🔓 Current Mode: PUBLIC"
      : "🔒 Current Mode: PRIVATE"
  );
});
