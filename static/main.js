document.addEventListener("DOMContentLoaded", () => {
    const resumeForm = document.getElementById("resumeForm");
    const relatedJobsForm = document.getElementById("relatedJobsForm");
    const resultContent = document.getElementById("resultContent");
    const relatedJobsContent = document.getElementById("relatedJobsContent");
    const chatMessages = document.getElementById("chatMessages");
    const userInput = document.getElementById("userInput");
    const sendButton = document.getElementById("sendButton");
    const navLinks = document.querySelectorAll(".nav-link");
    const sections = document.querySelectorAll(".content-section");
    const welcomeMessage = document.getElementById("welcome-message");
    const BASE_URL = ""; // Should be "" or your server address
    const urlParams = new URLSearchParams(window.location.search);
    const username = urlParams.get('username');
    if (username) {
        welcomeMessage.innerHTML = `Welcome, <span>${username}</span>!`;
    }
    navLinks.forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            const targetSectionId = e.target.getAttribute("data-section") + "-section";
            
            navLinks.forEach(nav => nav.classList.remove("active"));
            sections.forEach(sec => sec.classList.remove("active"));
            e.target.classList.add("active");
            document.getElementById(targetSectionId).classList.add("active");
        });
    });
    resumeForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const jobRole = resumeForm.jobRole.value.trim();
        const resumeFile = resumeForm.resume.files[0];
        if (!jobRole || !resumeFile) {
            alert("Please enter a job role and select a PDF resume.");
            return;
        }
        const formData = new FormData();
        formData.append("jobRole", jobRole);
        formData.append("resume", resumeFile);
        resultContent.innerHTML = "<em>Analyzing resume...</em>";
        try {
            const response = await fetch(`${BASE_URL}/upload`, {
                method: "POST",
                body: formData
            });
            // Check if response is JSON
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error('Server returned non-JSON response');
            }
            const data = await response.json();
            if (data.success) {
                resultContent.innerHTML = `
                    <div class='result-box'>
                        <h3>Analysis Result for <span>${data.jobRole}</span></h3>
                        <p><strong>Match Probability:</strong> ${data.probability}%</p>
                        <p><strong>Missing Skills:</strong> ${data.additionalSkills}</p>
                        <p><strong>Missing Frameworks:</strong> ${data.additionalFrameworks}</p>
                        <p><strong>Feedback:</strong> ${data.feedback}</p>
                    </div>
                `;
            } else {
                resultContent.innerHTML = `<span style='color:red;'>${data.error || 'Analysis failed.'}</span>`;
            }
        } catch (err) {
            console.error('Error:', err);
            resultContent.innerHTML = `<span style='color:red;'>Error: ${err.message}</span>`;
        }
    });
    relatedJobsForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const resumeFile = relatedJobsForm.relatedResume.files[0];
        if (!resumeFile) {
            alert("Please select a PDF resume to find related jobs.");
            return;
        }
        const formData = new FormData();
        formData.append("resume", resumeFile);
        relatedJobsContent.innerHTML = "<em>Finding related jobs...</em>";
        try {
            const response = await fetch(`${BASE_URL}/related_jobs`, {
                method: 'POST',
                body: formData
            });
            // Check if response is JSON
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error('Server returned non-JSON response');
            }
            const data = await response.json();
            if (data.success) {
                let html = '<h2>Top Job Matches</h2>';
                if (data.relatedJobs.length === 0) {
                    html += '<p>No job roles found in the database.</p>';
                } else {
                    html += '<div class="jobs-list">';
                    data.relatedJobs.forEach(job => {
                        let barColor = 'low';
                        if (job.probability >= 70) {
                            barColor = 'high';
                        } else if (job.probability >= 40) {
                            barColor = 'medium';
                        }
                        html += `
                            <div class="job-item">
                                <h3>${job.jobRole}</h3>
                                <p><strong>Match Probability:</strong> ${job.probability}%</p>
                                <div class="progress-bar-container">
                                    <div class="progress-bar" style="width: ${job.probability}%;" data-percentage="${barColor}"></div>
                                </div>
                                <p><strong>Missing Skills:</strong> ${job.additionalSkills}</p>
                                <p><strong>Missing Frameworks:</strong> ${job.additionalFrameworks}</p>
                            </div>
                        `;
                    });
                    html += '</div>';
                }
                relatedJobsContent.innerHTML = html;
            } else {
                relatedJobsContent.innerHTML = `<span style='color:red;'>${data.error || 'Finding jobs failed.'}</span>`;
            }
        } catch (err) {
            console.error('Error:', err);
            relatedJobsContent.innerHTML = `<span style='color:red;'>Error: ${err.message}</span>`;
        }
    });
    function appendMessage(sender, text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = sender === 'user' ? 'chat-msg user' : 'chat-msg bot';
        msgDiv.textContent = text;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    async function sendChat() {
        const message = userInput.value.trim();
        if (!message) return;
        appendMessage('user', message);
        userInput.value = '';
        appendMessage('bot', '...');
        try {
            const response = await fetch(`${BASE_URL}/chatbot`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message })
            });
            // Check if response is JSON
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error('Server returned non-JSON response');
            }
            const data = await response.json();
            // Remove the temporary "..." message
            chatMessages.removeChild(chatMessages.lastChild);
            if (data.success) {
                appendMessage('bot', data.response);
            } else {
                appendMessage('bot', data.error || 'No response.');
            }
        } catch (err) {
            console.error('Error:', err);
            chatMessages.removeChild(chatMessages.lastChild);
            appendMessage('bot', 'Error: ' + err.message);
        }
    }
    sendButton.addEventListener('click', sendChat);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendChat();
    });
});