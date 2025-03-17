// Beat Detection App - Upload Page JavaScript

// DOM Elements
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const fileInfo = document.getElementById('file-info');
const uploadProgress = document.getElementById('upload-progress');
const uploadForm = document.getElementById('upload-form');
const submitButton = document.getElementById('submit-button');

// Initialize the application
function init() {
    setupEventListeners();
}

// Set up all event listeners
function setupEventListeners() {
    // File selection via button
    fileInput.addEventListener('change', handleFileSelection);
    
    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    dropArea.addEventListener('drop', handleDrop, false);
    
    // Submit button click handler
    if (submitButton) {
        submitButton.addEventListener('click', handleSubmit);
    }
}

// Prevent default behaviors for drag and drop
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop area when dragging over
function highlight() {
    dropArea.classList.add('highlight');
}

// Remove highlight when dragging leaves
function unhighlight() {
    dropArea.classList.remove('highlight');
}

// Handle file drop
function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelection();
    }
}

// Handle file selection (via input or drop)
function handleFileSelection() {
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        
        // Check file type
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const validExtensions = ['mp3', 'wav', 'flac', 'm4a', 'ogg'];
        
        if (!validExtensions.includes(fileExtension)) {
            alert('Please select a valid audio file (MP3, WAV, FLAC, M4A, OGG)');
            fileInput.value = '';
            return;
        }
        
        // Display file info
        fileName.textContent = file.name;
        fileInfo.classList.remove('hidden');
        
        // IMPORTANT: This automatic upload is intentional and should not be removed.
        // It provides immediate feedback to users when they select a file.
        handleSubmit();
    }
}

// Handle form submission
function handleSubmit() {
    if (!fileInput.files.length) {
        alert('Please select a file first');
        return;
    }
    
    // Show upload progress
    fileInfo.classList.add('hidden');
    uploadProgress.classList.remove('hidden');
    setProgress(uploadProgress, 50);
    
    // Create FormData object
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('analyze', 'true');
    
    // Submit the form using fetch
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.redirected) {
            window.location.href = response.url;
        } else if (response.ok) {
            return response.json().then(data => {
                window.location.href = `/file/${data.file_id}`;
            });
        } else {
            return response.json().then(data => {
                throw new Error(data.detail || 'Upload failed');
            });
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error: ' + error.message);
        // Reset UI
        fileInfo.classList.remove('hidden');
        uploadProgress.classList.add('hidden');
        setProgress(uploadProgress, 0);
    });
}

// Set progress bar value
function setProgress(progressElement, value) {
    // If element doesn't exist, don't update
    if (!progressElement) {
        return;
    }
    
    const progressFill = progressElement.querySelector('.progress-fill');
    if (progressFill) {
        progressFill.style.width = `${value}%`;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);

// Add event listener for page unload to clean up resources
window.addEventListener('beforeunload', cleanup);

// Cleanup function to cancel any ongoing processes
function cleanup() {
    // Nothing specific to clean up in this simplified version
    // This is a placeholder for future cleanup needs
    console.log('Cleaning up resources before page unload');
}
