// Firebase Authentication Handler
// Uses modular SDK (imported from firebase-config.js)

import { 
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  updateProfile
} from "https://www.gstatic.com/firebasejs/12.8.0/firebase-auth.js";

// Get auth from global scope (exported in firebase-config.js)
// Access it as window.auth

// Switch between login and signup forms
function switchAuthForm() {
  const loginForm = document.getElementById('login-form');
  const signupForm = document.getElementById('signup-form');
  
  loginForm.classList.toggle('active');
  signupForm.classList.toggle('active');
}

// Make function globally accessible for onclick handlers
window.switchAuthForm = switchAuthForm;

// Real-time password validation - with error handling
document.addEventListener('DOMContentLoaded', () => {
  const passwordInput = document.getElementById('signup-password');
  if (passwordInput) {
    passwordInput.addEventListener('input', () => {
      const password = passwordInput.value;
      
      // Check each requirement
      const hasLength = password.length >= 8;
      const hasUppercase = /[A-Z]/.test(password);
      const hasLowercase = /[a-z]/.test(password);
      const hasNumber = /[0-9]/.test(password);
      
      // Update UI
      updateRequirement('req-length', hasLength);
      updateRequirement('req-uppercase', hasUppercase);
      updateRequirement('req-lowercase', hasLowercase);
      updateRequirement('req-number', hasNumber);
    });
  }
});

function updateRequirement(elementId, isMet) {
  const element = document.getElementById(elementId);
  if (element) {
    if (isMet) {
      element.classList.add('met');
    } else {
      element.classList.remove('met');
    }
  }
}

// Sign Up with Email/Password
document.getElementById('signup-btn').addEventListener('click', async () => {
  const name = document.getElementById('signup-name').value.trim();
  const email = document.getElementById('signup-email').value.trim();
  const password = document.getElementById('signup-password').value;
  const errorDiv = document.getElementById('signup-error');

  errorDiv.textContent = '';

  if (!name || !email || !password) {
    errorDiv.textContent = 'Please fill in all fields';
    return;
  }

  // Validate email format - MUST have @ and proper domain
  const emailRegex = /^[a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  if (!emailRegex.test(email)) {
    errorDiv.textContent = 'Invalid email format (e.g., user@gmail.com)';
    return;
  }

  // Extra validation: Check @ exists
  const atSymbolIndex = email.indexOf('@');
  if (atSymbolIndex === -1) {
    errorDiv.textContent = 'Email must contain @ symbol';
    return;
  }

  // Extra validation: Check there's text before @
  if (atSymbolIndex === 0) {
    errorDiv.textContent = 'Email must have a name before @';
    return;
  }

  // Extra validation: Check domain has dot
  const domain = email.substring(atSymbolIndex + 1);
  if (!domain.includes('.')) {
    errorDiv.textContent = 'Email domain must contain a dot (e.g., @gmail.com)';
    return;
  }

  // Extra validation: Check domain extension is valid
  const lastDotIndex = email.lastIndexOf('.');
  const tld = email.substring(lastDotIndex + 1);
  if (tld.length < 2) {
    errorDiv.textContent = 'Email domain extension must be at least 2 characters (e.g., .com)';
    return;
  }

  // Validate against common/known email domains
  const domainName = domain.toLowerCase();
  const validDomains = [
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'aol.com',
    'icloud.com', 'mail.com', 'protonmail.com', 'yandex.com', 'gmx.com',
    'zoho.com', 'fastmail.com', 'tutanota.com', 'mailbox.org', 'posteo.de',
    'inbox.com', 'mail.ru', 'live.com', 'msn.com', 'bellsouth.net',
    'comcast.net', 'verizon.net', 'cox.net', 'att.net', 'charter.net'
  ];

  // Check if domain is in the valid list
  if (!validDomains.includes(domainName)) {
    errorDiv.textContent = `Email domain "${domainName}" is not recognized. Please use a common email provider (e.g., gmail.com, yahoo.com, outlook.com)`;
    return;
  }

  // Validate password requirements
  if (password.length < 8) {
    errorDiv.textContent = 'Password must be at least 8 characters';
    return;
  }

  if (!/[A-Z]/.test(password)) {
    errorDiv.textContent = 'Password must contain at least 1 uppercase letter';
    return;
  }

  if (!/[a-z]/.test(password)) {
    errorDiv.textContent = 'Password must contain at least 1 lowercase letter';
    return;
  }

  if (!/[0-9]/.test(password)) {
    errorDiv.textContent = 'Password must contain at least 1 numeric number';
    return;
  }

  try {
    const userCredential = await createUserWithEmailAndPassword(window.auth, email, password);
    
    // Set user display name
    await updateProfile(userCredential.user, {
      displayName: name
    });

    console.log('User created:', userCredential.user);
    hideAuthModal();
  } catch (error) {
    console.error('Signup error:', error);
    errorDiv.textContent = error.message;
  }
});

// Sign In with Email/Password
document.getElementById('login-btn').addEventListener('click', async () => {
  const email = document.getElementById('login-email').value.trim();
  const password = document.getElementById('login-password').value;
  const errorDiv = document.getElementById('login-error');

  errorDiv.textContent = '';

  if (!email || !password) {
    errorDiv.textContent = 'Please enter email and password';
    return;
  }

  try {
    const userCredential = await signInWithEmailAndPassword(window.auth, email, password);
    console.log('User logged in:', userCredential.user);
    hideAuthModal();
  } catch (error) {
    console.error('Login error:', error);
    
    // User-friendly error messages
    if (error.code === 'auth/user-not-found' || error.code === 'auth/invalid-credential') {
      errorDiv.textContent = 'Invalid email or password';
    } else if (error.code === 'auth/invalid-email') {
      errorDiv.textContent = 'Invalid email format';
    } else if (error.code === 'auth/wrong-password') {
      errorDiv.textContent = 'Invalid email or password';
    } else if (error.code === 'auth/user-disabled') {
      errorDiv.textContent = 'This account has been disabled';
    } else {
      errorDiv.textContent = error.message;
    }
  }
});

// Sign Out
function handleLogout() {
  signOut(window.auth)
    .then(() => {
      console.log('User signed out');
      showAuthModal();
    })
    .catch((error) => console.error('Logout error:', error));
}

// Get all logout buttons and add event listeners
document.addEventListener('DOMContentLoaded', () => {
  const logoutButtons = [
    document.getElementById('logout-btn'),
    document.getElementById('logout-btn-edu'),
    document.getElementById('logout-btn-hist'),
    document.getElementById('logout-btn-set')
  ];

  logoutButtons.forEach(btn => {
    if (btn) {
      btn.addEventListener('click', handleLogout);
    }
  });
});

// Hide Auth Modal
function hideAuthModal() {
  const authModal = document.getElementById('auth-modal');
  authModal.classList.remove('active');
}

// Show Auth Modal
function showAuthModal() {
  const authModal = document.getElementById('auth-modal');
  authModal.classList.add('active');
}

// Monitor Authentication State
onAuthStateChanged(window.auth, (user) => {
  const authModal = document.getElementById('auth-modal');
  const userDisplay = [
    document.getElementById('user-display-name'),
    document.getElementById('user-display-name-edu'),
    document.getElementById('user-display-name-hist'),
    document.getElementById('user-display-name-set')
  ];
  
  const logoutButtons = [
    document.getElementById('logout-btn'),
    document.getElementById('logout-btn-edu'),
    document.getElementById('logout-btn-hist'),
    document.getElementById('logout-btn-set')
  ];

  if (user) {
    // User is logged in
    console.log('Current user:', user.email, user.displayName);
    
    const displayName = user.displayName || user.email.split('@')[0];
    
    // Update all user display elements
    userDisplay.forEach(el => {
      if (el) el.textContent = displayName;
    });

    // Show logout buttons
    logoutButtons.forEach(btn => {
      if (btn) btn.style.display = 'inline-block';
    });

    // Hide auth modal
    if (authModal) authModal.classList.remove('active');
    
    // Notify app about user change - load their chat history
    window.dispatchEvent(new CustomEvent('userChanged', { detail: { userId: user.uid } }));
  } else {
    // User is logged out
    console.log('No user logged in');
    
    // Reset display
    userDisplay.forEach(el => {
      if (el) el.textContent = 'Guest';
    });

    // Hide logout buttons
    logoutButtons.forEach(btn => {
      if (btn) btn.style.display = 'none';
    });

    // Show auth modal
    if (authModal) authModal.classList.add('active');
    
    // Clear all chat data when signing out
    clearAllChatData();
  }
});

// Function to clear all chat data (called on sign out)
function clearAllChatData() {
  // Clear chat messages from DOM
  const healthcareContainer = document.getElementById('healthcare-messages');
  const educationContainer = document.getElementById('education-messages');
  
  if (healthcareContainer) {
    healthcareContainer.innerHTML = '<div class="message bot">Please enter your medical inquiry.</div>';
  }
  if (educationContainer) {
    educationContainer.innerHTML = '<div class="message bot">Please enter your academic question.</div>';
  }
  
  // Clear chat history list
  const historyList = document.getElementById('history-list');
  if (historyList) {
    historyList.innerHTML = '<p style="color:#7f8c8d">No chat history found.</p>';
  }
  
  // Clear localStorage
  localStorage.removeItem('chatHistory');
  localStorage.removeItem('chatly-temperature');
  
  // Clear the history array in script.js by resetting it
  // This will be handled by a window event
  window.dispatchEvent(new Event('clearChatHistory'));
  
  console.log('All chat data cleared');
}
