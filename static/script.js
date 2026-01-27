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
      }, 0); // Changed to 0 for immediate switch
    });
  }

  // ---------- SIMPLE CHAT + HISTORY ----------
  const history = [];
  let currentUserId = null; // Track which user's data we're viewing

  function addMessage(chatId, sender, text) {
    console.log('Adding message:', chatId, sender, text);
    const container = document.getElementById(chatId + "-messages");
    console.log('Container:', container);
    if (!container) return;

    const msg = document.createElement("div");
    msg.className = "message " + sender;
    msg.innerHTML = text;

    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
  }

  function removeLastBotMessage() {
    const container = document.getElementById("healthcare-messages");
    if (!container) return;
    const messages = container.querySelectorAll(".message.bot");
    if (messages.length > 0) {
      container.removeChild(messages[messages.length - 1]);
    }
  }

  function updateLastBotMessage(newText) {
    const container = document.getElementById("healthcare-messages");
    if (!container) return;
    const messages = container.querySelectorAll(".message.bot");
    if (messages.length > 0) {
      messages[messages.length - 1].innerHTML = newText;
    }
  }

  // Listen for user change event (from Firebase auth)
  window.addEventListener('userChanged', (event) => {
    const userId = event.detail.userId;
    console.log('User changed to:', userId);
    currentUserId = userId;
    loadUserChatHistory(userId);
  });

  // Listen for clear chat history event (from Firebase auth - when switching users)
  window.addEventListener('clearChatHistory', () => {
    history.length = 0; // Clear the history array
    refreshHistory();
  });

  // Load chat history for specific user
  async function loadUserChatHistory(userId) {
    if (!userId) {
      history.length = 0;
      refreshHistory();
      return;
    }

    try {
      const response = await fetch(`/history?user_id=${encodeURIComponent(userId)}`);
      const result = await response.json();
      if (result.success) {
        history.length = 0;
        result.items.forEach(item => {
          history.push({
            topic: 'healthcare', // Assume healthcare for now
            text: item.question,
            timestamp: new Date(item.created_at)
          });
        });
        refreshHistory();

        // Rebuild messages on screen
        document.getElementById('healthcare-messages').innerHTML = '<div class="message bot">Upload a document first, then ask your questions.</div>';
        
        history.forEach(item => {
          addMessage(item.topic, 'user', item.text);
          // Don't add bot reply here, as it's saved in history
        });
      }
    } catch (error) {
      console.log('Could not load chat history from server');
      history.length = 0;
      refreshHistory();
    }
  }

  document.querySelectorAll(".chat-input-row").forEach((form) => {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      console.log('Form submitted, currentUserId:', currentUserId);

      const input = form.querySelector("input[type='text']");
      const submitBtn = form.querySelector("button[type='submit']");
      console.log('Submit button:', submitBtn);
      const text = input.value.trim();
      const fileInput = form.querySelector("input[type='file']");
      const file = fileInput ? fileInput.files[0] : null;

      console.log('Text:', text, 'File:', file);

      if (!text && !file) return;

      const topic = form.dataset.chat; // "healthcare"

      // Disable form during processing
      input.disabled = true;
      if (submitBtn) submitBtn.disabled = true;
      submitBtn.textContent = "Processing...";

      // If file is uploaded, send to upload endpoint first
      if (file) {
        console.log('Uploading file...');
        addMessage(topic, "bot", "Uploading and processing document... <span class='spinner'></span>");
        const formData = new FormData();
        formData.append('file', file);
        formData.append('user_id', currentUserId || 'guest');

        try {
          const uploadResponse = await fetch('/upload_docs', {
            method: 'POST',
            body: formData
          });
          const uploadResult = await uploadResponse.json();
          console.log('Upload result:', uploadResult);
          // Update the message with the result
          updateLastBotMessage(uploadResult.success ? uploadResult.message : `Upload failed: ${uploadResult.message}`);
          if (!uploadResult.success) {
            // Re-enable form
            input.disabled = false;
            if (submitBtn) submitBtn.disabled = false;
            submitBtn.textContent = "▶";
            return;
          }
        } catch (error) {
          console.log('Upload fetch error:', error);
          updateLastBotMessage(`Upload error: ${error.message}`);
          // Re-enable form
          input.disabled = false;
          if (submitBtn) submitBtn.disabled = false;
          submitBtn.textContent = "▶";
          return;
        }
      }

      if (text) {
        console.log('Sending query:', text);
        addMessage(topic, "user", text);

        // Send query to backend
        try {
          const response = await fetch('/query_rag', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text, user_id: currentUserId || 'guest' })
          });
          const result = await response.json();
          console.log('Query result:', result);
          if (result.success) {
            addMessage(topic, "bot", result.answer);
          } else {
            addMessage(topic, "bot", `Error: ${result.answer}`);
          }
        } catch (error) {
          console.log('Query fetch error:', error);
          addMessage(topic, "bot", `Error: ${error.message}`);
        }

        input.value = "";
      }

      if (file) {
        fileInput.value = ""; // Clear file input
      }

      // Re-enable form
      input.disabled = false;
      if (submitBtn) submitBtn.disabled = false;
      submitBtn.textContent = "▶";
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
