// Firebase Configuration
// Using Modular SDK (v12.8.0+)

import { initializeApp } from "https://www.gstatic.com/firebasejs/12.8.0/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/12.8.0/firebase-auth.js";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyB6Kum9Z9ppeHHxGscvsU7qzS0iCHyVDLs",
  authDomain: "chatly-25525.firebaseapp.com",
  projectId: "chatly-25525",
  storageBucket: "chatly-25525.firebasestorage.app",
  messagingSenderId: "531812892953",
  appId: "1:531812892953:web:247b95cf28328039170588"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Get auth reference - THIS CREATES THE auth OBJECT
const auth = getAuth(app);

// Export for use in other files
window.auth = auth;
