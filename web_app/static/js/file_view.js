// Beat Detection App - File View JavaScript

// DOM Elements
const analysisSection = document.getElementById('analysis-section');
const analysisProgress = document.getElementById('analysis-progress');
const analysisResults = document.getElementById('analysis-results');
const confirmButton = document.getElementById('confirm-button');
const cancelButton = document.getElementById('cancel-button');
const videoSection = document.getElementById('video-section');
const videoProgress = document.getElementById('video-progress');
const videoResults = document.getElementById('video-results');
const resultVideo = document.getElementById('result-video');
// Download button converted to a standard link - no longer need this reference
const restartButton = document.getElementById('restart-button');

// Result display elements
const resultBpm = document.getElementById('result-bpm');
const resultTotalBeats = document.getElementById('result-total-beats');
const resultDuration = document.getElementById('result-duration');
const resultMeter = document.getElementById('result-meter');

// Global state
let currentFileId = null;
let statusCheckInterval = null;

// Debug logging
let debugMode = false; // Set to true to enable debug panel
const debugMessages = [];

// Override console methods to capture logs
if (debugMode) {
    const originalLog = console.log;
    const originalWarn = console.warn;
    const originalError = console.error;
    
    console.log = function() {
        debugMessages.push({type: 'log', message: Array.from(arguments).join(' ')});
        updateDebugPanel();
        originalLog.apply(console, arguments);
    };
    
    console.warn = function() {
        debugMessages.push({type: 'warn', message: Array.from(arguments).join(' ')});
        updateDebugPanel();
        originalWarn.apply(console, arguments);
    };
    
    console.error = function() {
        debugMessages.push({type: 'error', message: Array.from(arguments).join(' ')});
        updateDebugPanel();
        originalError.apply(console, arguments);
    };
}

// Update the debug panel with latest messages
function updateDebugPanel() {
    if (!debugMode) return;
    
    const debugPanel = document.getElementById('debug-panel');
    if (!debugPanel) return;
    
    // Keep only the last 10 messages
    const recentMessages = debugMessages.slice(-10);
    
    // Clear and rebuild the debug panel
    debugPanel.innerHTML = '';
    recentMessages.forEach(msg => {
        const msgElement = document.createElement('div');
        msgElement.className = `debug-message debug-${msg.type}`;
        msgElement.textContent = msg.message;
        debugPanel.appendChild(msgElement);
    });
}

// Initialize the application
function init() {
    // Create debug panel if in debug mode
    if (debugMode) {
        const debugPanel = document.createElement('div');
        debugPanel.id = 'debug-panel';
        debugPanel.className = 'debug-panel';
        debugPanel.innerHTML = '<h3>Debug Messages</h3>';
        document.body.appendChild(debugPanel);
        
        // Add some basic styling
        const style = document.createElement('style');
        style.textContent = `
            .debug-panel {
                position: fixed;
                bottom: 0;
                right: 0;
                width: 400px;
                max-height: 300px;
                overflow-y: auto;
                background: rgba(0,0,0,0.8);
                color: white;
                padding: 10px;
                font-family: monospace;
                z-index: 9999;
                border-top-left-radius: 5px;
            }
            .debug-message {
                margin: 5px 0;
                padding: 5px;
                border-left: 3px solid #ccc;
            }
            .debug-log { border-color: #4CAF50; }
            .debug-warn { border-color: #FFC107; color: #FFC107; }
            .debug-error { border-color: #F44336; color: #F44336; }
        `;
        document.head.appendChild(style);
    }
    setupEventListeners();
    
    // Initialize with file data from the server
    if (window.initialFileData) {
        // Add very explicit debugging
        console.log('%c DEBUGGING INITIALIZATION', 'background: #222; color: #bada55; font-size: 20px;');
        console.log('Initial file data:', window.initialFileData);
        console.log('File ID:', window.initialFileData.fileId);
        console.log('Status object:', window.initialFileData.status);
        
        // Log beat and video tasks specifically
        const beatTask = window.initialFileData.status.beat_detection_task;
        const videoTask = window.initialFileData.status.video_generation_task;
        console.log('Beat detection task:', beatTask);
        console.log('Video generation task:', videoTask);
        
        // Log the specific properties we're checking
        if (beatTask) console.log('Beat task status:', beatTask.status);
        if (videoTask) console.log('Video task status:', videoTask.status);
        if (videoTask && videoTask.result) console.log('Video task result:', videoTask.result);
        
        currentFileId = window.initialFileData.fileId;
        
        // Force immediate check for the specific case of beat detection success with video generation failure
        const data = window.initialFileData.status;
        
        // Log DOM elements to verify they exist
        console.log('Analysis progress element:', analysisProgress);
        console.log('Analysis results element:', analysisResults);
        
        // Directly check and fix the issue regardless of task status
        // Handle any case where beat detection is SUCCESS, even if overall status is FAILURE
        if (data.status === 'ANALYZED' || data.status === 'FAILURE' || 
            (data.beat_detection_task && data.beat_detection_task.status === 'SUCCESS')) {
            console.log('%c FORCING DISPLAY OF ANALYSIS RESULTS', 'background: red; color: white; font-size: 16px;');
            console.log('Reason: Overall status:', data.status, 'Beat task status:', 
                data.beat_detection_task ? data.beat_detection_task.status : 'N/A');
            
            // Force hide analysis progress and show results
            if (analysisProgress) {
                console.log('Hiding analysis progress');
                analysisProgress.classList.add('hidden');
            }
            if (analysisResults) {
                console.log('Showing analysis results');
                analysisResults.classList.remove('hidden');
                displayAnalysisResults(data);
            }
        }
        
        // Handle different file states
        const status = window.initialFileData.status.status;
        
        if (status === 'ANALYZING') {
            // Show analysis section with progress
            analysisSection.classList.remove('hidden');
            analysisProgress.classList.remove('hidden');
            analysisResults.classList.add('hidden');
            
            // If we have progress information, update the progress bar
            const beatTask = window.initialFileData.status.beat_detection_task;
            if (beatTask && beatTask.progress) {
                // Handle both formats of progress data
                const progressValue = typeof beatTask.progress === 'object' && beatTask.progress.percent ? 
                    beatTask.progress.percent : beatTask.progress;
                updateProgressBar(analysisProgress.querySelector('.progress-fill'), progressValue);
            }
            
            // Start polling for status updates
            startStatusPolling();
        } 
        else if (status === 'ANALYZED') {
            // Show analysis results
            analysisSection.classList.remove('hidden');
            analysisProgress.classList.add('hidden');
            analysisResults.classList.remove('hidden');
            displayAnalysisResults(window.initialFileData.status);
        } 
        else if (status === 'GENERATING_VIDEO') {
            // Show both analysis results and video generation in progress
            analysisSection.classList.remove('hidden');
            analysisSection.classList.add('with-video');
            analysisProgress.classList.add('hidden');
            analysisResults.classList.remove('hidden');
            document.getElementById('analysis-buttons').classList.add('hidden');
            displayAnalysisResults(window.initialFileData.status);
            
            videoSection.classList.remove('hidden');
            videoProgress.classList.remove('hidden');
            videoResults.classList.add('hidden');
            
            // If we have progress information, update the progress bar
            const videoTask = window.initialFileData.status.video_generation_task;
            if (videoTask && videoTask.progress) {
                // Handle both formats of progress data
                const progressValue = typeof videoTask.progress === 'object' && videoTask.progress.percent ? 
                    videoTask.progress.percent : videoTask.progress;
                updateProgressBar(videoProgress.querySelector('.progress-fill'), progressValue);
            }
            
            // Start polling for status updates
            startStatusPolling();
        } 
        else if (status === 'COMPLETED') {
            // Show both analysis results and completed video
            analysisSection.classList.remove('hidden');
            analysisSection.classList.add('with-video');
            analysisProgress.classList.add('hidden');
            analysisResults.classList.remove('hidden');
            document.getElementById('analysis-buttons').classList.add('hidden');
            displayAnalysisResults(window.initialFileData.status);
            
            videoSection.classList.remove('hidden');
            videoProgress.classList.add('hidden');
            videoResults.classList.remove('hidden');
            displayVideo(window.initialFileData.status);
        }
        else if (status === 'ERROR') {
            // Show error message
            alert(`Error: ${window.initialFileData.status.error || 'Unknown error'}`);
        }
    } else {
        console.error('No file data provided');
    }
}

// Set up all event listeners
function setupEventListeners() {
    // Confirm button (generate video)
    confirmButton.addEventListener('click', handleConfirmAnalysis);
    
    // Cancel button
    cancelButton.addEventListener('click', handleCancelAnalysis);
    
    // Download button converted to a standard link - no longer need this event listener
    
    // Restart button
    restartButton.addEventListener('click', handleRestart);
}

// Start polling for status updates
function startStatusPolling() {
    // Clear any existing interval
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    
    // Check immediately
    checkStatus();
    
    // Set up interval for checking status
    statusCheckInterval = setInterval(checkStatus, 2000);
}

// Check the status of the current file
async function checkStatus() {
    if (!currentFileId) return;
    
    try {
        const response = await fetch(`/status/${currentFileId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Status update:', data);
        
        // Check for beat detection task
        const beatTask = data.beat_detection_task;
        // Check for video generation task
        const videoTask = data.video_generation_task;
        
        // Log the full structure to debug the stats issue
        console.log('Full data structure:', JSON.stringify(data, null, 2));
        
        // Handle status updates based on current state
        if (data.status === 'ANALYZED') {
            // Analysis completed
            
            // Get beat detection task data if available
            const beatTask = data.beat_detection_task || {};
            
            // Check if we have valid stats in the beat detection task
            const hasValidStats = beatTask && beatTask.stats && Object.keys(beatTask.stats).length > 0;
            
            if (hasValidStats) {
                // We have valid stats, so we can stop polling and display the results
                console.log('Valid stats found, displaying analysis results');
                analysisProgress.classList.add('hidden');
                analysisResults.classList.remove('hidden');
                displayAnalysisResults(data);
                
                // Only clear interval if we're not waiting for video generation
                if (!data.video_generation_task || 
                    (data.video_generation_task.status !== 'STARTED' && 
                     data.video_generation_task.status !== 'PROGRESS')) {
                    clearInterval(statusCheckInterval);
                }
            } else {
                // This is an unusual case - the server says ANALYZED but we don't have stats
                console.warn('Status is ANALYZED but no valid stats found. Beat task status:', 
                    beatTask ? beatTask.status : 'undefined');
                
                // If beat detection task is SUCCESS but no stats, show results anyway
                // This handles cases where the task succeeded but stats weren't properly returned
                if (beatTask && beatTask.status === 'SUCCESS') {
                    console.log('Beat detection task is SUCCESS, showing results despite missing stats');
                    analysisProgress.classList.add('hidden');
                    analysisResults.classList.remove('hidden');
                    displayAnalysisResults(data);
                    
                    // Only clear interval if we're not waiting for video generation
                    if (!data.video_generation_task || 
                        (data.video_generation_task.status !== 'STARTED' && 
                         data.video_generation_task.status !== 'PROGRESS')) {
                        clearInterval(statusCheckInterval);
                    }
                } else {
                    // Continue polling only if we're still waiting for something
                    console.warn('Continuing to poll for complete results...');
                }
            }
        }
        else if (data.status === 'GENERATING_VIDEO' && videoSection.classList.contains('hidden')) {
            // Video generation started
            document.getElementById('analysis-buttons').classList.add('hidden');
            analysisSection.classList.add('with-video');
            videoSection.classList.remove('hidden');
            videoProgress.classList.remove('hidden');
            videoResults.classList.add('hidden');
            // Keep polling for updates
        }
        else if (data.status === 'COMPLETED' && videoProgress.classList.contains('hidden') === false) {
            // Video generation completed
            videoProgress.classList.add('hidden');
            videoResults.classList.remove('hidden');
            displayVideo(data);
            clearInterval(statusCheckInterval);
        }
        else if (data.status === 'ERROR' || data.status === 'FAILURE') {
            // Error occurred
            clearInterval(statusCheckInterval);
            
            // Always check if beat detection succeeded, regardless of video task status
            if (beatTask && beatTask.status === 'SUCCESS') {
                console.log('%c ERROR/FAILURE with successful beat detection', 'background: blue; color: white;');
                // Beat detection succeeded, make sure to show results
                analysisProgress.classList.add('hidden');
                analysisResults.classList.remove('hidden');
                displayAnalysisResults(data);
            }
            
            // Check if this is a video generation error
            // Need to handle both FAILURE status and SUCCESS status with error result
            if (videoTask) {
                console.log('Video task in error handler:', videoTask);
                // Hide video progress regardless of specific error type
                videoProgress.classList.add('hidden');
                
                // Make sure video section is visible
                videoSection.classList.remove('hidden');
                videoResults.classList.remove('hidden');
                
                // Create error message element if it doesn't exist
                let errorElement = document.getElementById('video-error-message');
                if (!errorElement) {
                    errorElement = document.createElement('div');
                    errorElement.id = 'video-error-message';
                    errorElement.className = 'error-message';
                    videoResults.appendChild(errorElement);
                }
                
                // Get error message from the appropriate location
                let errorMsg;
                if (videoTask.status === 'SUCCESS' && videoTask.result) {
                    // For tasks that returned error results
                    errorMsg = videoTask.result.error || 'Video generation failed';
                } else {
                    // For tasks that failed with exceptions
                    errorMsg = videoTask.error || data.error || 'Video generation failed';
                }
                
                errorElement.textContent = `Error: ${errorMsg}`;
                errorElement.style.display = 'block';
                
                console.error('Video generation error:', errorMsg);
            } else {
                // General error or beat detection error
                alert(`Error: ${data.error || 'Unknown error'}`);
            }
        }
        // Update progress indicators if tasks are in progress
        else if (beatTask && (beatTask.status === 'STARTED' || beatTask.status === 'PROGRESS') && beatTask.progress) {
            // Handle both formats of progress data
            const progressValue = typeof beatTask.progress === 'object' && beatTask.progress.percent ? 
                beatTask.progress.percent : beatTask.progress;
            updateProgressBar(analysisProgress.querySelector('.progress-fill'), progressValue);
        }
        else if (videoTask && (videoTask.status === 'STARTED' || videoTask.status === 'PROGRESS') && videoTask.progress) {
            // Handle both formats of progress data
            const progressValue = typeof videoTask.progress === 'object' && videoTask.progress.percent ? 
                videoTask.progress.percent : videoTask.progress;
            updateProgressBar(videoProgress.querySelector('.progress-fill'), progressValue);
        }
        
    } catch (error) {
        console.error('Error checking status:', error);
    }
}

// Display analysis results
function displayAnalysisResults(data) {
    console.log('Displaying analysis results with data:', JSON.stringify(data, null, 2));
    
    // Get beat detection task data if available
    const beatTask = data.beat_detection_task || {};
    console.log('Beat detection task:', JSON.stringify(beatTask, null, 2));
    
    // Only look for stats inside the beat detection task object
    let stats = {};
    if (beatTask && beatTask.stats) {
        stats = beatTask.stats;
        console.log('Found stats in beatTask.stats');
    }
    
    console.log('Final stats object used:', JSON.stringify(stats, null, 2));
    
    // Check if we have valid stats
    if (Object.keys(stats).length === 0) {
        console.warn('No stats found in the beat detection task yet. This might be a race condition.');
        console.log('Will continue polling for status updates until stats are available.');
        // Don't return - we'll just show placeholder values and continue polling
    }
    
    // Update the UI with the stats values
    resultBpm.textContent = stats.bpm ? Number(stats.bpm).toFixed(1) : '--';
    resultTotalBeats.textContent = stats.total_beats || '--';
    resultDuration.textContent = stats.duration ? formatTime(stats.duration) : '--';
    resultMeter.textContent = stats.detected_meter || '--';
    
    // No need to fetch beat data for visualization anymore
}

// Display the generated video
function displayVideo(data) {
    // Set video source to use the download endpoint instead of the non-existent video endpoint
    resultVideo.src = `/download/${currentFileId}`;
    resultVideo.load();
    
    // Also make sure to display the analysis results
    // This ensures stats are shown even when coming directly to the completed video
    displayAnalysisResults(data);
}

// Handle confirm analysis button click
async function handleConfirmAnalysis() {
    if (!currentFileId) return;
    
    try {
        // Disable the button to prevent multiple clicks
        const confirmButton = document.getElementById('confirm-button');
        if (confirmButton) {
            confirmButton.disabled = true;
            confirmButton.textContent = 'Processing...';
        }
        
        // Log current status for debugging
        console.log('Attempting to confirm analysis for file:', currentFileId);
        
        // Now try to confirm the analysis
        const response = await fetch(`/confirm/${currentFileId}`, {
            method: 'POST',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Confirmation response:', data);
        
        // Keep analysis section visible and also show video section
        // Hide the generate video button to prevent multiple clicks
        document.getElementById('analysis-buttons').classList.add('hidden');
        // Add the with-video class to the analysis section for better styling
        analysisSection.classList.add('with-video');
        videoSection.classList.remove('hidden');
        videoProgress.classList.remove('hidden');
        videoResults.classList.add('hidden');
        
        // Start polling for status updates
        startStatusPolling();
        
    } catch (error) {
        console.error('Error confirming analysis:', error);
        alert(`Error starting video generation: ${error.message}`);
        
        // Re-enable the button
        const confirmButton = document.getElementById('confirm-button');
        if (confirmButton) {
            confirmButton.disabled = false;
            confirmButton.textContent = 'Generate Video';
        }
    }
}

// Handle cancel analysis button click
async function handleCancelAnalysis() {
    if (!currentFileId) return;
    
    if (confirm('Are you sure you want to cancel? This will delete the current analysis.')) {
        try {
            const response = await fetch(`/cancel/${currentFileId}`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            // Redirect to home page
            window.location.href = '/';
            
        } catch (error) {
            console.error('Error canceling analysis:', error);
            alert('Error canceling analysis. Please try again.');
        }
    }
}

// Download button converted to a standard link - function no longer needed

// Handle restart button click
function handleRestart() {
    // Redirect to home page
    window.location.href = '/';
}

// Format seconds to MM:SS
function formatTime(seconds) {
    if (typeof seconds !== 'number') return '--:--';
    
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

// Update progress bar with percentage
function updateProgressBar(progressElement, percentage) {
    if (progressElement && typeof percentage === 'number') {
        // Ensure percentage is between 0 and 100
        const validPercentage = Math.min(100, Math.max(0, percentage));
        progressElement.style.width = `${validPercentage}%`;
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);
