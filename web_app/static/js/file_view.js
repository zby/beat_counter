// Beat Detection App - File View JavaScript

/*
POLLING LOGIC:
1. When polling starts:
   - During initialization (init function):
     - When status is 'ANALYZING' - polling starts
     - When status is 'GENERATING_VIDEO' - polling starts
   - During user actions:
     - When user clicks "Generate Video" button (handleConfirmAnalysis function) - polling starts

2. When polling stops:
   - When status is 'ANALYZED' with valid stats and no video generation in progress - polling stops
   - When status is 'COMPLETED' - polling stops
   - When status is 'ERROR', 'ANALYZING_FAILURE', or 'VIDEO_ERROR' - polling stops

3. Error states (no polling):
   - When status is 'ANALYZED' but no valid stats are found - this is an error state, polling does not start
*/

// Global state
let currentFileId = null;
let statusCheckInterval = null;
// Set debugMode to false by default, can be overridden in browser console if needed
let debugMode = document.getElementById('debug-panel') ? true : false;

// --- Helper function to safely add/remove 'hidden' class ---
const setVisibility = (element, shouldBeVisible) => {
    if (element) {
        if (shouldBeVisible) {
            element.classList.remove('hidden');
        } else {
            element.classList.add('hidden');
        }
    } else {
        // console.warn("Attempted to set visibility for a non-existent element.");
    }
};

// --- Helper function to format numbers with optional decimal places ---
const formatNumber = (value, decimalPlaces = 0) => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value !== 'number') return value;
    return value.toFixed(decimalPlaces);
};

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing Beat Detection App File View');

    // Check for initialFileData passed from the template
    if (window.initialFileData && window.initialFileData.fileId) {
        console.log('Using window.initialFileData:', window.initialFileData);
        initWithFileData(window.initialFileData);
    } else {
        console.error('No initialFileData available. Cannot initialize application.');
        // Optionally display an error message to the user on the page
        const mainContent = document.querySelector('main');
        if(mainContent) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message card';
            errorDiv.textContent = 'Error initializing page: File data missing.';
            mainContent.innerHTML = ''; // Clear existing content
            mainContent.appendChild(errorDiv);
        }
    }
});

// Initialize app with file data
function initWithFileData(fileData) {
    // Validate file ID before setting it
    const fileId = fileData.fileId;
    if (!fileId || typeof fileId !== 'string') {
        console.error('Invalid file ID format:', fileId);
        // Display error on page?
        return;
    }

    // Set current file ID for use in API calls
    currentFileId = fileId;

    // Initialize UI based on current status
    // Pass the full data object which now contains flattened results
    handleStatus(fileData);

    // Set up event listeners for buttons etc.
    setupEventListeners();

    // Set up debug panel if enabled and element exists
    if (debugMode && document.getElementById('debug-panel')) {
        setupDebugPanel();
    } else {
        // Ensure debug panel is hidden if not enabled or element missing
        const debugPanel = document.getElementById('debug-panel');
        if (debugPanel) debugPanel.style.display = 'none';
    }
}

// Set up event listeners
function setupEventListeners() {
    // Confirm button - Generate Video
    const confirmButton = document.getElementById('confirm-button');
    if (confirmButton) {
        confirmButton.addEventListener('click', function() {
            if (!currentFileId) {
                 console.error("Cannot confirm analysis: fileId is missing.");
                 return;
            }

            console.log(`Confirming analysis for file: ${currentFileId}`);
            // Disable the button immediately to prevent double clicks
            confirmButton.disabled = true;
            confirmButton.textContent = 'Generating...'; // Provide visual feedback

            // Send request to confirm analysis
            fetch(`/confirm/${currentFileId}`, {
                method: 'POST',
                headers: {
                     'Content-Type': 'application/json',
                     'X-Requested-With': 'XMLHttpRequest' // Indicate AJAX
                }
            })
            .then(response => {
                // Check for non-OK HTTP status codes first
                if (!response.ok) {
                    // Attempt to parse error JSON, otherwise use status text
                    return response.json().catch(() => {
                         // If JSON parsing fails, create a generic error
                         throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
                    }).then(errorData => {
                         // Throw an error with the detail message from backend
                         throw new Error(errorData.detail || `HTTP error ${response.status}`);
                    });
                }
                // If response is OK (200), parse JSON
                return response.json();
            })
            .then(data => {
                 console.log('Confirmation successful:', data);
                 
                 // Instead of manually updating the state, start polling the file status
                 // The next poll will return GENERATING_VIDEO status
                 startStatusPolling();
                 
                 // Update UI to show we're starting video generation
                 const inProgressUI = {
                     ...window.initialFileData,
                     status: 'GENERATING_VIDEO'
                 };
                 handleStatus(inProgressUI);
            })
            .catch(error => {
                console.error('Error confirming analysis:', error);
                alert('Error starting video generation: ' + error.message);
                // Re-enable the button on error
                confirmButton.disabled = false;
                confirmButton.textContent = 'Generate Video';
            });
        });
    }

    // Cancel button (now acts as "Upload New File" or similar)
    const cancelButton = document.getElementById('cancel-button');
    if (cancelButton) {
        cancelButton.addEventListener('click', function() {
            // Redirect to the home page (upload form)
            window.location.href = '/';
        });
    }

    // Restart button (in completed section)
    const restartButton = document.getElementById('restart-button');
    if (restartButton) {
        restartButton.addEventListener('click', function() {
            // Redirect to the home page (upload form)
            window.location.href = '/';
        });
    }
}

// Handle different file statuses to control UI visibility
function handleStatus(statusData) {
    if (!statusData || !statusData.status) {
        console.error('No status data or status string provided to handleStatus');
        return;
    }

    const status = statusData.status;
    console.log('Handling status:', status);
    // Update debug panel if enabled
    if (debugMode && window.updateDebugPanel) {
         window.updateDebugPanel(); // Update with latest data potentially modified
    }


    // Get references to all relevant DOM elements
    const analysisSection = document.getElementById('analysis-section');
    const analysisProgress = document.getElementById('analysis-progress');
    const analysisResults = document.getElementById('analysis-results');
    const analysisButtons = document.getElementById('analysis-buttons'); // Contains confirm/cancel
    const videoSection = document.getElementById('video-section');
    const videoProgress = document.getElementById('video-progress');
    const videoResults = document.getElementById('video-results'); // Contains video player and download/restart buttons
    const analysisErrorMsgContainer = document.getElementById('analysis-error-message'); // Dedicated container
    const videoErrorMsgContainer = document.getElementById('video-error-message');       // Dedicated container

    // --- Clear previous error messages ---
    if (analysisErrorMsgContainer) analysisErrorMsgContainer.innerHTML = '';
    if (videoErrorMsgContainer) videoErrorMsgContainer.innerHTML = '';
    setVisibility(analysisErrorMsgContainer, false);
    setVisibility(videoErrorMsgContainer, false);


    // --- Default visibility (assume nothing is shown initially) ---
    // Analysis Section
    setVisibility(analysisSection, true); // Always show the analysis card wrapper
    setVisibility(analysisProgress, false);
    setVisibility(analysisResults, false);
    setVisibility(analysisButtons, false);
    // Video Section
    setVisibility(videoSection, false); // Hide video card wrapper by default
    setVisibility(videoProgress, false);
    setVisibility(videoResults, false);

    // --- Process based on status ---
    switch (status) {
        case 'ANALYZING':
            console.log('UI State: ANALYZING');
            setVisibility(analysisProgress, true); // Show analysis progress bar
            setVisibility(analysisResults, false);
            setVisibility(analysisButtons, false);
            setVisibility(videoSection, false); // Hide entire video section

            // Start polling for status updates
            startStatusPolling();

            // Update the progress bar if we have progress data
            if (statusData.task_progress) {
                updateProgressBar('analysis', statusData.task_progress);
            }
            break;

        case 'ANALYZED':
            console.log('UI State: ANALYZED');
            setVisibility(analysisProgress, false); // Hide progress
            setVisibility(analysisResults, true);   // Show results grid
            setVisibility(analysisButtons, true);    // Show confirm/cancel buttons
            setVisibility(videoSection, false);      // Hide entire video section

            displayAnalysisResults(statusData); // Populate results grid

            // Stop polling as analysis is done
            stopPolling();
            break;

        case 'ANALYZING_FAILURE':
            console.log('UI State: ANALYZING_FAILURE');
            setVisibility(analysisProgress, false); // Hide progress
            setVisibility(analysisResults, true);  // Show results section (to contain error)
            setVisibility(analysisButtons, false); // Hide confirm/cancel buttons

            // Display specific error message in the analysis section
            if (analysisErrorMsgContainer) {
                 analysisErrorMsgContainer.textContent = statusData.beat_error
                      ? `Error during beat analysis: ${statusData.beat_error}`
                      : 'Error: Beat analysis failed.';
                 setVisibility(analysisErrorMsgContainer, true);
            } else {
                 // Fallback if dedicated container doesn't exist
                 displayGenericError(analysisResults, 'Error: Beat analysis failed.');
            }

            setVisibility(videoSection, false); // Hide entire video section

            // Stop polling as task has failed
            stopPolling();
            break;

        case 'GENERATING_VIDEO':
            console.log('UI State: GENERATING_VIDEO');
            setVisibility(analysisProgress, false);  // Hide analysis progress
            setVisibility(analysisResults, true);   // Show analysis results (was successful)
            setVisibility(analysisButtons, false);   // Hide confirm/cancel buttons
            setVisibility(videoSection, true);       // Show video section wrapper
            setVisibility(videoProgress, true);    // Show video progress bar
            setVisibility(videoResults, false);      // Hide final video results/buttons

            displayAnalysisResults(statusData); // Populate results grid

            // Start polling for status updates
            startStatusPolling();

            // Update the progress bar if we have progress data
            if (statusData.task_progress) {
                updateProgressBar('video', statusData.task_progress);
            }
            break;

        case 'COMPLETED':
            console.log('UI State: COMPLETED');
            setVisibility(analysisProgress, false);  // Hide analysis progress
            setVisibility(analysisResults, true);   // Show analysis results
            setVisibility(analysisButtons, false);   // Hide confirm/cancel buttons
            setVisibility(videoSection, true);       // Show video section wrapper
            setVisibility(videoProgress, false);     // Hide video progress bar
            setVisibility(videoResults, true);       // Show final video results/buttons

            displayAnalysisResults(statusData); // Populate results grid
            displayVideo();                     // Load and show the video

            // Stop polling as process is complete
            stopPolling();
            break;

        case 'VIDEO_ERROR':
            console.log('UI State: VIDEO_ERROR');
            // --- FIX START ---
            setVisibility(analysisProgress, false); // Hide analysis progress
            setVisibility(analysisResults, true);  // Show analysis results (was successful)
            setVisibility(analysisButtons, false);  // Hide confirm/cancel buttons

            setVisibility(videoSection, true);      // Show video section wrapper
            setVisibility(videoProgress, false);    // Hide video progress bar
            setVisibility(videoResults, true);     // Show video results section (to contain error message & potentially retry/cancel options)

            displayAnalysisResults(statusData); // Show the successful beat analysis results

            // Display specific error message in the video section
             if (videoErrorMsgContainer) {
                 videoErrorMsgContainer.textContent = statusData.video_error
                     ? `Error during video generation: ${statusData.video_error}`
                     : 'Error: Video generation failed.';
                 setVisibility(videoErrorMsgContainer, true);

                 // Hide the video player element itself since there's no video
                 const videoPlayer = document.getElementById('result-video');
                 setVisibility(videoPlayer, false);
                 // Consider adjusting buttons in videoResults if needed (e.g., hide download, show retry?)
                 const downloadButton = videoResults.querySelector('a.primary-button');
                 setVisibility(downloadButton, false); // Hide download button on error


            } else {
                 // Fallback if dedicated container doesn't exist
                 displayGenericError(videoResults, 'Error: Video generation failed.');
            }
            // --- FIX END ---

            // Stop polling as task has failed
            stopPolling();
            break;

        case 'ERROR':
        default: // Handle unexpected or generic ERROR status
            console.error('UI State: ERROR or Unknown - ', status);
            // Show a generic error message, potentially hide everything else
            setVisibility(analysisProgress, false);
            setVisibility(analysisResults, true); // Show analysis section to contain error
            setVisibility(analysisButtons, false);
            setVisibility(videoSection, false); // Hide video section

            displayGenericError(analysisResults, 'An unexpected error occurred during processing.');

            // Stop polling
            stopPolling();
            break;
    }
}

// Helper to display a generic error message within a container
function displayGenericError(containerElement, message) {
    if (containerElement) {
        // Clear previous content maybe? Or just prepend error.
        // containerElement.innerHTML = ''; // Clear container
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message'; // Use existing CSS class
        errorDiv.textContent = message;
        // Prepend so it appears at the top of the section
        containerElement.prepend(errorDiv);
        setVisibility(containerElement, true); // Make sure container is visible
    }
}


// Display analysis results using flattened data structure
function displayAnalysisResults(data) {
    // Update result display elements only if they exist
    const setText = (id, value, suffix = '', defaultValue = 'N/A') => {
        const element = document.getElementById(id);
        if (element) {
            // Check for null/undefined/0 before formatting
            const displayValue = (value !== null && value !== undefined && value !== 0)
                               ? (typeof value === 'number' ? value.toFixed(1) : value) + suffix
                               : defaultValue;
            element.textContent = displayValue;
        }
    };

    // Get beat stats values, checking if they exist
    const beatStats = data.beat_stats || {};
    const totalBeats = beatStats.total_beats || data.totalBeats;
    const bpm = beatStats.tempo_bpm || data.bpm;
    const beatsPerBar = beatStats.detected_beats_per_bar || data.detectedBeatsPerBar;
    const irregularityPercent = beatStats.irregularity_percent || data.irregularityPercent;

    setText('result-total-beats', formatNumber(totalBeats));
    setText('result-beats-per-bar', beatsPerBar); // Assuming beatsPerBar is not a number needing formatting
    setText('result-bpm', formatNumber(bpm, 1) + ' BPM');
    setText('result-duration', formatNumber(data.duration, 1) + 's');
    setText('result-irregularity', formatNumber(irregularityPercent, 1) + '%');
}


// Display video - load source and update links
function displayVideo() {
    console.log('displayVideo called for file:', currentFileId);
    if (!currentFileId) {
        console.error("Cannot display video: currentFileId is not set.");
        return;
    }

    const resultVideo = document.getElementById('result-video');
    if (!resultVideo) {
        console.error('Video player element (#result-video) not found');
        return;
    }
    setVisibility(resultVideo, true); // Ensure player is visible

    const videoSourceUrl = `/download/${currentFileId}`;
    console.log('Setting video source to:', videoSourceUrl);

    // Set the video source only if it's different or not set
    if (resultVideo.currentSrc !== videoSourceUrl) {
        resultVideo.src = videoSourceUrl;
        resultVideo.load(); // Explicitly load the new source
        console.log('Video source set and load() called');
    } else {
         console.log('Video source already set.');
         // Ensure it tries to play if already loaded
         resultVideo.play().catch(err => console.warn('Video play failed (maybe requires user interaction):', err));
    }


    // Add event listeners for debugging (consider removing in production)
    resultVideo.removeEventListener('loadstart', handleVideoEvent); // Remove previous listeners first
    resultVideo.removeEventListener('loadeddata', handleVideoEvent);
    resultVideo.removeEventListener('canplay', handleVideoEvent);
    resultVideo.removeEventListener('error', handleVideoErrorEvent);
    resultVideo.addEventListener('loadstart', handleVideoEvent);
    resultVideo.addEventListener('loadeddata', handleVideoEvent);
    resultVideo.addEventListener('canplay', handleVideoEvent);
    resultVideo.addEventListener('error', handleVideoErrorEvent);


    // Update download link
    const downloadLink = document.querySelector('#video-results a.primary-button[download]');
    if (downloadLink) {
        downloadLink.href = videoSourceUrl;
        setVisibility(downloadLink, true); // Ensure download button is visible
        console.log('Download link updated');
    }

    // Make sure video container is visible (handled in handleStatus now)
    // const videoResultsContainer = document.getElementById('video-results');
    // setVisibility(videoResultsContainer, true);
}

// --- Video Event Handlers (for debugging) ---
function handleVideoEvent(e) {
    console.log(`Video event: ${e.type}`);
}
function handleVideoErrorEvent(e) {
    const videoElement = e.target;
    let errorMsg = 'Unknown video error';
    if (videoElement.error) {
        switch (videoElement.error.code) {
            case videoElement.error.MEDIA_ERR_ABORTED:
                errorMsg = 'Video playback aborted.';
                break;
            case videoElement.error.MEDIA_ERR_NETWORK:
                errorMsg = 'Video download failed due network error.';
                break;
            case videoElement.error.MEDIA_ERR_DECODE:
                errorMsg = 'Video playback failed due to decoding error.';
                break;
            case videoElement.error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                errorMsg = 'Video source not supported or not found.';
                break;
            default:
                errorMsg = 'An unknown error occurred during video playback.';
                break;
        }
    }
    console.error(`Video error event: ${e.type} - Code: ${videoElement.error?.code} - Message: ${errorMsg}`);
    // Optionally display this error to the user more prominently
    const videoErrorContainer = document.getElementById('video-error-message');
    if(videoErrorContainer) {
         videoErrorContainer.textContent = `Video Error: ${errorMsg}`;
         setVisibility(videoErrorContainer, true);
    }

}


// --- Polling Logic ---

// Stop polling for status updates
function stopPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
        console.log('Polling stopped.');
    }
}

// Start polling for status updates
function startStatusPolling() {
    // Clear any existing interval first
    stopPolling();

    if (!currentFileId) {
        console.warn('startStatusPolling called without a currentFileId.');
        return;
    }

    console.log(`Polling started for file: ${currentFileId}`);
    // Set up new polling interval (every 3 seconds)
    statusCheckInterval = setInterval(checkFileStatus, 3000);

    // Do an immediate check first
    checkFileStatus();
}

// Check the current status of the file
function checkFileStatus() {
    if (!currentFileId) {
        console.log('No file ID available for status check, stopping polling.');
        stopPolling();
        return;
    }

    console.log(`Checking status for file: ${currentFileId}`);

    // Fetch the current file status from the centralized status endpoint
    fetch(`/status/${currentFileId}`)
        .then(response => {
            if (!response.ok) {
                // If file not found (404), stop polling gracefully
                if (response.status === 404) {
                    console.warn(`File ${currentFileId} not found, stopping polling.`);
                    stopPolling();
                    // Update UI to an error state
                    handleStatus({ ...window.initialFileData, status: 'ERROR' });
                    return null; // Indicate no data to process
                }
                // For other errors, throw to be caught below
                throw new Error(`Failed to fetch file status: ${response.statusText}`);
            }
            return response.json();
        })
        .then(statusData => {
            if (statusData === null) return; // Stop processing if file wasn't found

            console.log('Received file status:', statusData);

            // Update global data with latest status info
            window.initialFileData = {
                ...window.initialFileData,
                ...statusData
            };

            // Update UI based on the new status data
            handleStatus(statusData);

            // Determine if we should stop polling
            const terminalStates = ['ANALYZED', 'COMPLETED', 'ERROR', 'ANALYZING_FAILURE', 'VIDEO_ERROR'];
            if (terminalStates.includes(statusData.status)) {
                console.log(`File ${currentFileId} reached terminal state: ${statusData.status}. Stopping polling.`);
                stopPolling();
            } else {
                console.log(`File ${currentFileId} still processing (Status: ${statusData.status}). Polling continues.`);
            }
        })
        .catch(error => {
            console.error('Error during file status check:', error);
            // Consider stopping polling on persistent errors
            // For now, just log the error and continue polling
        });
}

// Update progress bar based on task progress data
function updateProgressBar(type, progressData) {
    // Determine which progress bar to update based on type
    const progressBarId = type === 'video' ? 'video-progress' : 'analysis-progress';
    const progressContainer = document.getElementById(progressBarId);
    if (!progressContainer) return; // Exit if the container isn't visible or found

    const progressFill = progressContainer.querySelector('.progress-fill');
    const progressText = progressContainer.querySelector('p');
    if (!progressFill) return; // Exit if inner elements are missing

    // Extract progress value and status message from progressData
    let progressPercent = 0;
    let progressMessage = '';

    if (progressData && typeof progressData === 'object') {
        progressPercent = progressData.percent || 0;
        progressMessage = progressData.status || ''; // Get status message
    } else if (progressData === 'STARTED') {
        // Provide a default message if only state is available
        progressPercent = 5; // Small progress indication
        progressMessage = 'Starting...';
    } else if (progressData === 'PENDING') {
        progressPercent = 0;
        progressMessage = 'Waiting in queue...';
    }

    // Clamp percentage between 0 and 100
    progressPercent = Math.max(0, Math.min(100, progressPercent));

    // Update the progress bar width
    progressFill.style.width = `${progressPercent}%`;

    // Update progress text if available
    if (progressText) {
        if (progressMessage) {
            progressText.textContent = progressMessage; // Display the status message
        } else {
            // Fallback message if no detailed status is available
            progressText.textContent = `${type === 'video' ? 'Generating video' : 'Analyzing beats'}... (${progressPercent.toFixed(0)}%)`;
        }
    }
}


// --- Debug Panel ---

// Setup debug panel
function setupDebugPanel() {
    const debugPanel = document.getElementById('debug-panel');
    if (!debugPanel) return;

    // Basic styling (consider moving to CSS)
    debugPanel.style.display = 'block';
    debugPanel.style.backgroundColor = 'rgba(0,0,0,0.8)';
    debugPanel.style.color = '#eee';
    debugPanel.style.padding = '15px';
    debugPanel.style.margin = '20px 0';
    debugPanel.style.borderRadius = '8px';
    debugPanel.style.fontFamily = 'monospace';
    debugPanel.style.fontSize = '12px';
    debugPanel.style.lineHeight = '1.4';
    debugPanel.style.maxHeight = '400px';
    debugPanel.style.overflow = 'auto';
    debugPanel.style.whiteSpace = 'pre-wrap'; // Wrap long lines
    debugPanel.style.wordBreak = 'break-all'; // Break long words/IDs

    // Expose update function globally for console access if needed
    window.updateDebugPanel = updateDebugPanel;
    // Initial update
    updateDebugPanel();
}
// Update debug panel with current state information
function updateDebugPanel() {
    const debugPanel = document.getElementById('debug-panel');
    // Ensure debugMode is checked and panel exists
    if (!debugMode || !debugPanel) return;

    // Use the potentially updated global data
    const data = window.initialFileData || {};

    // Build debug content using template literals for readability
    const debugContent = `
### Debug Information (${new Date().toLocaleTimeString()}) ###

File ID:       ${currentFileId || 'N/A'}
Overall Status: ${data.status || 'N/A'}
Polling Active: ${statusCheckInterval ? 'YES' : 'NO'}

--- Config ---
Duration Limit: ${data.duration_limit || data.durationLimit || 'N/A'}s
Original Dur:   ${data.original_duration?.toFixed(2) || data.originalDuration?.toFixed(2) || 'N/A'}s
App Dir:        ${data.app_dir || data.appDir || 'N/A'}

--- Analysis ---
BPM:            ${data.beat_stats?.tempo_bpm?.toFixed(1) || data.bpm?.toFixed(1) || 'N/A'}
Total Beats:    ${data.beat_stats?.total_beats || data.totalBeats || 'N/A'}
Audio Dur:      ${data.duration?.toFixed(2) || 'N/A'}s
Beats per bar:  ${data.beat_stats?.detected_beats_per_bar || data.detectedBeatsPerBar || 'N/A'}

--- Tasks ---
Beat Task Progress: ${JSON.stringify(data.beat_task_progress || {})}
Beat Error:         ${data.beat_error || 'None'}

Video Task Progress: ${JSON.stringify(data.video_task_progress || {})}
Video Error:         ${data.video_error || 'None'}

--- Raw Data ---
${JSON.stringify(data, null, 2)}
    `;

    // Display in debug panel
    debugPanel.textContent = debugContent.trim(); // Use textContent for pre-like formatting
}