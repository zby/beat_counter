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
    
    // Submit the form
    uploadForm.submit();
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
