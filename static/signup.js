document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.getElementById("loginForm");
    const registerForm = document.getElementById("registerForm");
    const container = document.querySelector(".container");
    const registerBtn = document.querySelector('.register-btn');
    const loginBtn = document.querySelector('.login-btn');
    const BASE_URL = window.location.hostname === 'localhost' 
        ? 'http://localhost:5000' 
        : 'https://jobsinline.onrender.com';

    loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const username = loginForm.querySelector("input[name='username']").value;
        const password = loginForm.querySelector("input[name='password']").value;

        try {
            const response = await fetch(`${BASE_URL}/auth`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "login", username, password })
            });

            const data = await response.json();
            alert(data.message);

            if (data.redirect === "register") {
                container.classList.add("active");
                loginForm.reset();
            } else if (data.redirect === "main") {
                window.location.href = `/main.html?username=${encodeURIComponent(data.username)}`;
            }
        } catch (error) {
            alert("Error during login: " + error.message);
        }
    });

    registerForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const username = registerForm.querySelector("input[name='username']").value;
        const email = registerForm.querySelector("input[name='email']").value;
        const password = registerForm.querySelector("input[name='password']").value;
        const phone = registerForm.querySelector("input[name='phone']").value;

        try {
            const response = await fetch(`${BASE_URL}/auth`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "register", username, email, phone, password })
            });

            const data = await response.json();
            alert(data.message);

            if (data.success) {
                container.classList.remove("active");
                registerForm.reset();
            }
        } catch (error) {
            alert("Error during registration: " + error.message);
        }
    });

    registerBtn.addEventListener('click', () => {
        container.classList.add('right-panel-active');
        loginForm.reset();
    });

    loginBtn.addEventListener('click', () => {
        container.classList.remove('right-panel-active');
        registerForm.reset();
    });
});