document.addEventListener("DOMContentLoaded", () => {
  // ---------- SCREEN SWITCHING ----------
  const screens = document.querySelectorAll(".screen");

  // track which chat screen was last active
  let lastChatScreen = "screen-healthcare";
  let selectedTopic = "healthcare";

  function showScreen(id) {
    // remember last chat screen
    if (id === "screen-healthcare" || id === "screen-education") {
      lastChatScreen = id;
    }

    screens.forEach((s) => s.classList.toggle("active", s.id === id));

    // highlight nav items
    document.querySelectorAll(".nav-item").forEach((item) => {
      const target = item.dataset.target;
      item.classList.toggle("active", target === id);
    });
  }

  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => {
      let target = item.dataset.target;
      const label = item.textContent.trim();

      // If the nav item is "Your Chat", go back to whichever chat was last used
      if (label === "Your Chat") {
        target = lastChatScreen;
      }

      if (target) showScreen(target);
    });
  });

  // ---------- CLICK LOGO/TITLE TO GO HOME ----------
  const homeBtn = document.getElementById("home-button");
  if (homeBtn) {
    homeBtn.addEventListener("click", () => {
      showScreen("screen-home");
    });
  }

  // ---------- TOPIC SELECTION ----------
  document.querySelectorAll(".topic-circle").forEach((btn) => {
    btn.addEventListener("click", () => {
      selectedTopic = btn.dataset.topic || "healthcare";
      document.querySelectorAll(".topic-circle").forEach((b) =>
        b.classList.remove("selected")
      );
      btn.classList.add("selected");
    });
  });

  // ---------- START CHAT + CUSTOM LOADING ----------
  const startBtn = document.getElementById("btn-start-chat");
  const loadingTextEl = document.getElementById("loading-text");

  if (startBtn && loadingTextEl) {
    startBtn.addEventListener("click", () => {
      const targetScreen =
        selectedTopic === "healthcare"
          ? "screen-healthcare"
          : "screen-education";

      // set loading text based on topic
      if (selectedTopic === "healthcare") {
        loadingTextEl.textContent = "Scrubbing in...";
      } else {
        loadingTextEl.textContent = "Organising your resources...";
      }

      // show loading first
      showScreen("screen-loading");

      // fake loading delay, then jump to chat
      setTimeout(() => {
        showScreen(targetScreen);
      }, 2000);
    });
  }

  // ---------- SIMPLE CHAT + HISTORY ----------
  const history = [];
  let currentUserId = null; // Track which user's data we're viewing

  function addMessage(chatId, sender, text) {
    const container = document.getElementById(chatId + "-messages");
    if (!container) return;

    const msg = document.createElement("div");
    msg.className = "message " + sender;
    msg.textContent = text;

    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
  }

  // Listen for user change event (from Firebase auth)
  window.addEventListener('userChanged', (event) => {
    const userId = event.detail.userId;
    currentUserId = userId;
    loadUserChatHistory(userId);
  });

  // Listen for clear chat history event (from Firebase auth - when switching users)
  window.addEventListener('clearChatHistory', () => {
    history.length = 0; // Clear the history array
    refreshHistory();
  });

  // Load chat history for specific user
  function loadUserChatHistory(userId) {
    if (!userId) {
      history.length = 0;
      return;
    }

    // Load from localStorage using user ID as key
    const savedHistory = localStorage.getItem(`chatHistory-${userId}`);
    history.length = 0; // Clear current history
    
    if (savedHistory) {
      try {
        const parsedHistory = JSON.parse(savedHistory);
        history.push(...parsedHistory);
      } catch (e) {
        console.log('Could not parse chat history');
      }
    }
    
    refreshHistory();
    
    // Rebuild messages on screen
    document.getElementById('healthcare-messages').innerHTML = '<div class="message bot">Please enter your medical inquiry.</div>';
    document.getElementById('education-messages').innerHTML = '<div class="message bot">Please enter your academic question.</div>';
    
    history.forEach(item => {
      addMessage(item.topic, 'user', item.text);
      setTimeout(() => {
        addMessage(
          item.topic,
          'bot',
          `Thanks for your question about ${item.topic}. (demo reply)`
        );
      }, 100);
    });
  }

  document.querySelectorAll(".chat-input-row").forEach((form) => {
    form.addEventListener("submit", (e) => {
      e.preventDefault();

      const input = form.querySelector("input");
      const text = input.value.trim();
      if (!text) return;

      const topic = form.dataset.chat; // "healthcare" or "education"

      addMessage(topic, "user", text);

      history.push({
        topic,
        text,
        timestamp: new Date(),
      });

      // Save to localStorage with user ID key
      if (currentUserId) {
        localStorage.setItem(`chatHistory-${currentUserId}`, JSON.stringify(history));
      }

      refreshHistory();

      // demo AI reply
      setTimeout(() => {
        addMessage(
          topic,
          "bot",
          `Thanks for your question about ${topic}. (demo reply)`
        );
      }, 400);

      input.value = "";
    });
  });

  function refreshHistory() {
    const list = document.getElementById("history-list");
    if (!list) return;

    list.innerHTML = "";

    if (!history.length) {
      list.innerHTML = `<p style="color:#7f8c8d">No chat history found.</p>`;
      return;
    }

    history
      .slice()
      .reverse()
      .forEach((item) => {
        const div = document.createElement("div");
        div.className = "history-item";
        div.innerHTML = `
          ${item.text}
          <small style="color:#aaa; float:right;">${item.topic}</small>
        `;
        list.appendChild(div);
      });
  }

  refreshHistory();

  // ---------- TEMPERATURE SLIDER (0–8) + LIVE VALUE + LABEL ----------
  const tempSlider = document.querySelector('[data-setting="temperature"]');
  const tempValue = document.getElementById("temp-value");

  function getTempLabel(value) {
    value = Number(value);

    if (value <= 1) return "Strict";
    if (value <= 3) return "Safe";
    if (value === 4) return "Balanced";
    if (value <= 6) return "Creative";
    return "Very Creative";
  }

  if (tempSlider && tempValue) {
    const savedTemp = localStorage.getItem("chatly-temperature");

    // Load saved temperature or fallback to slider's default
    const initialValue = savedTemp !== null ? savedTemp : tempSlider.value;
    tempSlider.value = initialValue;
    tempValue.textContent = `Value: ${initialValue} — ${getTempLabel(initialValue)}`;

    // Update text + save when user moves the slider
    tempSlider.addEventListener("input", () => {
      const val = tempSlider.value;
      tempValue.textContent = `Value: ${val} — ${getTempLabel(val)}`;
      localStorage.setItem("chatly-temperature", val);
      console.log("Temperature set to:", val);
    });
  }

  // ---------- DARK MODE TOGGLE (light/dark theme switch) ----------
  const darkToggle = document.querySelector('[data-setting="darkmode"]');

  if (darkToggle) {
    // Helper to apply theme + sync toggle
    const applyTheme = (makeLight) => {
      if (makeLight) {
        document.body.classList.add("light-mode");
        darkToggle.classList.add("active");
        localStorage.setItem("chatly-theme", "light");
      } else {
        document.body.classList.remove("light-mode");
        darkToggle.classList.remove("active");
        localStorage.setItem("chatly-theme", "dark");
      }
    };

    // 1. Load saved theme (if any)
    const saved = localStorage.getItem("chatly-theme");
    if (saved === "light") {
      applyTheme(true);
    } else {
      applyTheme(false); // default = dark (your current theme)
    }

    // 2. When user clicks the toggle
    darkToggle.addEventListener("click", () => {
      const willBeLight = !document.body.classList.contains("light-mode");
      applyTheme(willBeLight);
    });
  }
});
