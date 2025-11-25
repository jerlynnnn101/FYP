document.addEventListener("DOMContentLoaded", () => {
  // ---------- SCREEN SWITCHING ----------
  const screens = document.querySelectorAll(".screen");

  function showScreen(id) {
    screens.forEach((s) => s.classList.toggle("active", s.id === id));

    // highlight nav items
    document.querySelectorAll(".nav-item").forEach((item) => {
      const target = item.dataset.target;
      item.classList.toggle("active", target === id);
    });
  }

  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => {
      const target = item.dataset.target;
      if (target) showScreen(target);
    });
  });

  // ---------- TOPIC SELECTION ----------
  let selectedTopic = "healthcare";

  document.querySelectorAll(".topic-circle").forEach((btn) => {
    btn.addEventListener("click", () => {
      selectedTopic = btn.dataset.topic;
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

  function addMessage(chatId, sender, text) {
    const container = document.getElementById(chatId + "-messages");
    if (!container) return;

    const msg = document.createElement("div");
    msg.className = "message " + sender;
    msg.textContent = text;

    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
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
