/*!
* Start Bootstrap - Grayscale v7.0.6 (https://startbootstrap.com/theme/grayscale)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-grayscale/blob/master/LICENSE)
*/
// --- IMPORTANT: All client-side code for dynamic content loading, ---
// --- year calculation, and hardcoded tech lists has been removed, ---
// --- as this logic is now handled by Flask/Jinja2 on the server. ---

// ----------------------------------------------------------------------
// -------------------- GLOBAL THEME COLORS -----------------------------
// ----------------------------------------------------------------------
const darkTextColor = '#E5E7EB'; // Light gray/white for text
const darkGridColor = 'rgba(255, 255, 255, 0.2)'; // Faint white grid lines
const barWeightedColor = 'hsla(181, 71%, 71%, 0.80)'; // Teal/Cyan color (Used for CV bars)
const barRawColor = 'rgba(191, 193, 194, 0.7)'; // Light gray muted color (Not used here, but kept for reference)
const darkBgColor = '#1F2937'; // Slate 800 (For plot background)

const staticSelectors = '#iabout, #icontact, #itop';
const dynamicSelectors = window.CONTENT_SECTION_IDS.map(tech => `#${tech}`).join(', ');
const allSelectors = staticSelectors + ', ' + dynamicSelectors;

/* ---------------------------------------------------------------------- */
/*                               sleep                                    */
/* ---------------------------------------------------------------------- */
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/* ---------------------------------------------------------------------- */
/*                     dynamic window height                              */
/* ---------------------------------------------------------------------- */
function setMainContentHeight() {
    // 1. Get the full viewport height
    const viewportHeight = window.innerHeight; 
    // 2. Get the actual height of the fixed header and footer
    const headerHeight = document.getElementById('header').offsetHeight;
    const footerHeight = document.getElementById('footer').offsetHeight;
    // 3. Get the main content element
    const mainContent = document.getElementById('main-content');
    // 4. Calculate the required height
    // Height = Viewport - Header - Footer
    const calculatedHeight = viewportHeight - headerHeight - footerHeight - 16;
    // 5. Apply the calculated height
    mainContent.style.height = calculatedHeight + 'px';
    console.log(`Viewport: ${viewportHeight}px, Header: ${headerHeight}px, Footer: ${footerHeight}px, Main Content set to: ${calculatedHeight}px`);
}
window.onload = setMainContentHeight;                     // Run the calculation on page load
window.addEventListener('resize', setMainContentHeight);  // Run the calculation every time the window is resized

/* ---------------------------------------------------------------------- */
/* CUSTOM SCROLLSPY USING INTERSECTION OBSERVER (THE FIX)                 */
/* ---------------------------------------------------------------------- */
function initializeScrollspyObserver() {
    const scrollContainer = document.getElementById('scroll-container'); // The element with the scrollbar
    const navLinks = document.querySelectorAll('.mylink'); // All links in your nav
    const activeDisplay = document.getElementById('active-section-display');

    if (!scrollContainer || navLinks.length === 0) {
        console.warn('ScrollSpy Observer initialization failed: Scroll container or navigation links not found.');
        return;
    }

    // Function to handle activation/deactivation
    const setActiveLink = (id) => {
        // 1. Update the link highlighting
        navLinks.forEach(link => link.classList.remove('active'));

        // Find and add active class to the corresponding link
        const targetLink = document.querySelector(`.mylink[href="#${id}"]`);
        if (targetLink) {
            targetLink.classList.add('active');
        }

        // 2. Update the Debug Display (THIS IS THE KEY ADDITION)
        if (activeDisplay) {
            activeDisplay.textContent = `Active: ${id}`;
        }
    };

    // Track the currently active section
    let currentActiveId = 'itop';

    // Intersection Observer Callback
    const observerCallback = (entries) => {
        let activeEntry = null;

        // 1. Collect all currently intersecting entries
        entries.forEach(entry => {
            // Check if the element is currently intersecting our detection zone
            if (entry.isIntersecting) {
                // If this is the first one found, or if it is physically highest up
                // than the currently stored active entry, set it as the new active entry.
                if (activeEntry === null || entry.boundingClientRect.top < activeEntry.boundingClientRect.top) {
                    activeEntry = entry;
                }
            }
        });

        // 2. Update the active link if a new highest-intersecting element is found
        if (activeEntry) {
            // Only update the DOM if the active ID has actually changed
            if (currentActiveId !== activeEntry.target.id) {
                currentActiveId = activeEntry.target.id;
                setActiveLink(currentActiveId);
                // ##############################################################################################
                // ##################################      kontrola            ##################################
                // ##############################################################################################
                document.getElementById("myText").innerHTML = " -> " + currentActiveId.toLowerCase();
                // document.getElementById("myText").innerHTML = allSelectors;
                // ##############################################################################################
            }
        }

        // Fallback for when scrolling to the very top (scrollTop is 0)
        if (scrollContainer.scrollTop < 10) {
             // Only update if it's not already 'itop' to avoid unnecessary DOM updates
             if (currentActiveId !== 'itop') {
                 currentActiveId = 'itop';
                 setActiveLink('itop');
             }
        }
    };
    // Intersection Observer Options
    const observerOptions = {
        root: scrollContainer,
        // Define a narrow detection zone (the "trigger line") at the top of the scroll container.
        // -90% bottom margin creates a 10% detection zone at the top.
        rootMargin: '0px 0px -80% 0px',
        threshold: 0 // Fires immediately when an element enters/leaves the detection zone
    };

    const observer = new IntersectionObserver(observerCallback, observerOptions);

    // 4. Observe all target sections
    // Note: This relies on the section IDs matching the link href attributes (e.g., #aboutx)
    document.querySelectorAll(allSelectors).forEach(section => {
        observer.observe(section);
    });

    // 5. Initial state check: Set 'itop' as active on load
    setActiveLink('itop');
}

// ---------------------------------------------------------------------
// ------    runFibonacciTest --- /api/run_fibonacci_test   ----- PYTHON
// ---------------------------------------------------------------------
async function runFibonacciTest() {
    const button = document.getElementById('runFibonacciTestBtn');
    const input = document.getElementById('fibonacciNInput');
    const outputArea = document.getElementById('fibonacciOutput');
    const n = input.value;
    const apiUrl = '/api/run_fibonacci_test';
    const textDecoder = new TextDecoder();  // new

    // Reset UI state - Clear previous output and show loading
    outputArea.innerHTML = '<b>SYSTEM:</b> Starting connection...\n';
    outputArea.classList.remove('text-danger');
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> running ... ';

    try {
        const response = await smartFetch(apiUrl, {    // await fetch replaced with await smartFetch 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n: n })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server returned status ${response.status}: ${errorText}`);
        }
        
        // Start reading the stream from the response body
        const reader = response.body.getReader();
        let buffer = ''; // Buffer to hold incomplete data chunks

        while (true) {
            // Read the next chunk from the stream
            const { value, done } = await reader.read();
            
            if (done) {
                // If the stream is done, process any remaining buffer
                if (buffer.trim().length > 0) {
                    processBuffer(buffer);
                }
                break;
            }

            // Convert chunk (Uint8Array) to string and append to buffer
            buffer += textDecoder.decode(value, { stream: true });
            
            // Process the buffer line by line, looking for complete JSON objects
            // The Flask generator yields complete JSON objects terminated by '\n'
            let lastNewlineIndex = buffer.lastIndexOf('\n');
            if (lastNewlineIndex !== -1) {
                // Extract complete lines
                const completeData = buffer.substring(0, lastNewlineIndex + 1);
                // Keep the remainder in the buffer
                buffer = buffer.substring(lastNewlineIndex + 1);
                processBuffer(completeData);
            }
        }
        
        // Function to parse and display the received data chunks
        function processBuffer(dataChunk) {
            // Split the chunk by newline, filter out empty lines
            const lines = dataChunk.trim().split('\n').filter(line => line.trim() !== '');
            lines.forEach(line => {
                try {
                    const parsedData = JSON.parse(line);
                    // Append the message to the terminal output
                    outputArea.innerHTML += parsedData.message;
                    // Auto-scroll to the bottom of the terminal
                    outputArea.scrollTop = outputArea.scrollHeight; 
                } catch (e) {
                    console.error('Failed to parse JSON line:', line, e);
                    outputArea.textContent += `ERROR: Corrupted stream data received.\n`;
                }
            });
        }

    } catch (error) {
        outputArea.textContent += `CRITICAL ERROR: Network or Server issue.\n// Details: ${error.message}`;
        outputArea.classList.add('text-danger');
        console.error('Fibonacci API streaming failed:', error);
    } finally {
        // Hide loading and re-enable button
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-play me-2"></i><span class="text-uppercase">Run</span> fibonacci.py';

    }
}    

// ---------------------------------------------------------------------
// ---    runFibonacciPytest --- /api/run_fibonacci_pytest   ---- PYTEST
// ---------------------------------------------------------------------
async function runFibonacciPytest() {
    const outputEl = document.getElementById('fibonacciOutput');
    const btn = document.getElementById('runFibonacciPytestBtn');
    const PYTEST_ENDPOINT = '/api/run_fibonacci_pytest';

    if (!outputEl || !btn) return;

    // 1. Reset UI and disable button
    outputEl.innerHTML = `Running Pytest suite...`;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> running ... ';
    
    try {
        const response = await smartFetch(PYTEST_ENDPOINT, {    // await fetch replaced with await smartFetch 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
            const errorData = await response.json();
            outputEl.innerHTML = `Error: ${errorData.error}`;
            return;
        }
        const data = await response.json();
        
        // 2. Display the pytest output
        // Pytest output is returned as a single string from the Flask endpoint.
        if (data.pytest_output) {
            // Use <pre> tag to maintain whitespace and line breaks from the terminal output
            outputEl.innerHTML = `TERMINAL OUTPUT:\n${data.pytest_output}`;
        } else {
             outputEl.innerHTML = `Error: Pytest returned no output.`;
        }

    } catch (error) {
        outputEl.innerHTML = `Network Error: ${error.message}`;
        console.error('Fetch error:', error);
    } finally {
        // 3. Re-enable button
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play me-2"></i><span class="text-uppercase">Run</span> test_fibonacci.py';
    }
}

// ---------------------------------------------------------------------
// --------------    fetchMovies --- /api/fetch_movies   -----  REST API
// ---------------------------------------------------------------------
async function fetchMovies() {
    const button = document.getElementById('fetchMoviesBtn');
    const resultsContainer = document.getElementById('movieResults');   
    const statusMessageDiv = document.getElementById('statusMessage');
    const API_ENDPOINT = '/api/fetch_movies';

    if (!button || !resultsContainer) {
        console.error("API demo elements not found.");
        return;
    }

    // Set loading state 
    statusMessageDiv.textContent = 'Checking cache...';
    statusMessageDiv.className = 'text-sm font-medium text-gray-500 p-2 rounded-lg bg-gray-100';
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> running ... ';
    resultsContainer.innerHTML = '<p class="text-indigo-500 italic">Loading data...</p>';
    // resultsContainer.innerHTML = '<p class="text-center text-muted m-0">Fetching data from simulated backend...</p>';

    await sleep(1000);  // 2. Wait for 1000ms (1 second)
    try {
      const response = await smartFetch(API_ENDPOINT, {    // await fetch replaced with await smartFetch 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            const data = await response.json();

            // --- PROCESS AND DISPLAY CACHE STATUS ---
            if (data.cache_status) {
                let html1 =`Cache Status: ${data.cache_status}` ;
                html1 += '<br>';
                html1 += `(The cache is updated from the TMDB API once every ${data.expiration} hours.)`;
                statusMessageDiv.innerHTML = html1;
                // statusMessageDiv.textContent = `Cache Status: ${data.cache_status}` ;
                // `(The cache is updated from the API once every ${data.expiration} hours.)`;
                
                // Simple color coding based on the status message
                statusMessageDiv.classList.remove('text-gray-500', 'text-green-600', 'text-yellow-600', 'text-red-600', 'bg-gray-100');
                
                if (data.cache_status.includes('SUCCESS: Data loaded')) {
                    statusMessageDiv.classList.add('text-green-800', 'bg-green-100');
                } else if (data.cache_status.includes('expired') || data.cache_status.includes('empty')) {
                    statusMessageDiv.classList.add('text-yellow-800', 'bg-yellow-100');
                } else if (data.cache_status.includes('FAILED') || data.cache_status.includes('Error')) {
                    statusMessageDiv.classList.add('text-red-800', 'bg-red-100');
                } else {
                    statusMessageDiv.classList.add('text-gray-500', 'bg-gray-100');
                }
            }

            // //--- MOCK API RESPONSE ---
            // const mockData = {movies: [
            //         { title: "The Shawshank Redemption", year: 1994, score: 9.3 },
            //         { title: "The Godfather", year: 1972, score: 9.2 },
            //         { title: "The Dark Knight", year: 2008, score: 9.0 },
            //         { title: "Schindler's List", year: 1993, score: 8.9 },
            //         { title: "xThe Lord of the Rings: The Return of the King", year: 2003, score: 8.9 },
            // ]};
            // // Simulate 1.5s network delay
            // await new Promise(resolve => setTimeout(resolve, 1500)); 
            // const data = mockData;

            let html = '<h5 class="text-success mb-3"><b>Top 5 Movies from TMDB:</b></h5>';
            html += '<div class="px-2">';
            data.movies.forEach((movie, index) => {
                html += `
                    <div class="movie-item">
                        <span class="fw-bold me-2 text-primary">${index + 1}.</span> 
                        ${movie.title} <span class="text-muted small">(${movie.year})</span>
                        <span class="float-end badge bg-primary">${movie.score} / 10</span>
                    </div>
                `;
            });
            html += '</div>';
            resultsContainer.innerHTML = html;
        } else {
            const errorData = await response.json();
        // Handle server-side errors
            resultsContainer.innerHTML = `<p class="text-red-500">Error: ${data.error || 'Unknown server error.'}</p>`;
        }
    } catch (error) {
        console.error('Fetch error:', error);
        resultsContainer.innerHTML = `<p class="text-red-500">Failed to connect to server: ${error.message}</p>`;
        statusMessageDiv.textContent = 'Connection Error';
        statusMessageDiv.classList.remove('bg-gray-100');
        statusMessageDiv.classList.add('text-red-800', 'bg-red-100');
    } finally {
        // 3. Re-enable button
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-play me-2"></i> Get TMDB Top 5 Movies';
    }
}

// Global variable to hold the chart instance so we can destroy and redraw it
let weightedScoreChartInstance = null;
let plotlyChartInstance = null;

// ----------------------------------------------------------------------
// ----------------- UTILITY FUNCTION FOR RENDERING -------------- PANDAS
// ----------------------------------------------------------------------
function formatNumber(num) {
    // Formats numbers with commas (e.g., 10000 -> 10,000)
    if (typeof num === 'number') {
        return num.toLocaleString(undefined, { maximumFractionDigits: 0 });
    }
    return String(num);
}

// ----------------------------------------------------------------------
// -------------------- CHART RENDERING FUNCTION ----------------- PANDAS
// ----------------------------------------------------------------------
/**
 * Renders a horizontal bar chart comparing Raw Average Score vs. Weighted Score
 * for the top 5 movies using Chart.js.
 * @param {Array<Object>} topMovies - Array of movie objects with scores and titles.
 */
function renderWeightedScoreChart(topMovies) {
    const ctx = document.getElementById('weightedScoreChart');
    if (!ctx) return;

    // Destroy existing chart instance if it exists to prevent overlap/memory leak
    if (weightedScoreChartInstance) {
        weightedScoreChartInstance.destroy();
    }

    // Map data for Chart.js
    const titles = topMovies.map(m => `${m.title} (${m.release_year})`);
    const rawScores = topMovies.map(m => parseFloat(m.vote_average.toFixed(2)));
    const weightedScores = topMovies.map(m => parseFloat(m.weighted_score.toFixed(3)));

    // Create the new chart instance
    weightedScoreChartInstance = new Chart(ctx, {
        type: 'bar',

        data: {
            labels: titles,
            datasets: [
                {
                    label: 'Weighted Score',
                    data: weightedScores,
                    backgroundColor: barWeightedColor, 
                    borderColor: barWeightedColor, 
                    borderWidth: 1,
                },
                {
                    label: 'Raw Average Score',
                    data: rawScores,
                    backgroundColor: barRawColor, 
                    borderColor: barRawColor,
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y', // Renders bars horizontally
            // Global chart options for dark theme
            color: darkTextColor, // This sets the default font color for the whole chart

            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: darkTextColor // Legend label color
                    }
                },
                title: {
                    display: true,
                    text: 'Top 5 Movie Score Comparison (Raw vs. Weighted)',
                    color: darkTextColor, // Title text color
                    font: { size: 16, weight: 'bold'}
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += context.parsed.x.toFixed(3);
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Score (Max 10.0)',
                        color: darkTextColor // X-axis title color
                    },
                    min: 7.5, // Start the scale higher to emphasize differences
                    max: 10.0,
                    ticks: {
                        color: darkTextColor // X-axis tick labels color
                    },
                    grid: {
                        color: darkGridColor // X-axis grid lines color
                    }
                },
                y: {
                    ticks: {
                        color: darkTextColor // Y-axis tick labels color
                    },
                    grid: {
                        color: darkGridColor // Y-axis grid lines color
                    }
                }
            }
        }
    });
}

// ----------------------------------------------------------------------
// --------------------- PANDAS ANALYSIS RUNNER ------------------ PANDAS
// ----------------------------------------------------------------------
async function runPandasAnalysis() {
    const button = document.getElementById('runPandasAnalysisBtn');
    const statusDiv = document.getElementById('pandasStatus');
    const pandasContainer = document.getElementById('pandasResults');  

    // Disable button and show spinner
    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> running ... ';
    pandasContainer.innerHTML = '<p class="text-indigo-500 italic">Loading data...</p>';

    await sleep(1000);  // 2. Wait for 1000ms (1 second)

    try {
        // Call the new Flask endpoint that performs the Pandas processing
        const response = await smartFetch('/api/get_pandas_data', {    // await fetch replaced with await smartFetch 
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();

        if (response.ok) {
            // statusDiv.className = 'alert alert-success';
            statusDiv.innerHTML = `<strong>SUCCESS!</strong> Analysis complete. Source Status: <em>${result.source_status}</em>`;

            // 1. Display Summary Statistics
            const stats = result.summary_stats;
            let html = `
            <div class="mt-3 row">
                <h4 class="text-xl font-semibold mb-3">Analysis Summary</h4>
                <div class="row row-cols-2 row-cols-md-4 text-center">
                    <div class="col">
                        <div class="p-3 bg-dark rounded shadow-sm border">
                            <h5 class="text-2xl font-bold text-cyan">${stats.total_movies_analyzed}</h5>
                            <p class="text-sm">Total Movies Analyzed</p>
                        </div>
                    </div>
                    <div class="col">
                        <div class="p-3 bg-dark rounded shadow-sm border">
                            <h5 class="text-2xl font-bold text-cyan">${stats.mean_score}</h5>
                            <p class="text-sm">Overall Mean Score</p>
                        </div>
                    </div>

                    <div class="col">
                        <div class="p-3 bg-dark rounded shadow-sm border">
                            <h5 class="text-2xl font-bold text-cyan">${stats.min_votes_for_qualification}</h5>
                            <p class="text-sm">Min Votes (75th Percentile)</p>
                        </div>
                    </div>
                    <div class="col">
                        <div class="p-3 bg-dark rounded shadow-sm border">
                            <h5 class="text-2xl font-bold text-cyan">${stats.median_votes}</h5>
                            <p class="text-sm">Median Vote Count</p>
                        </div>
                    </div>
                </div>
            </div>
            `;
            // 2. Display CHART AREA
            html += `
                <h4 class="text-xl font-semibold mt-5 mb-3">Weighted Score Visualization <span class="text-success text-xl">(a.k.a. "Bayesian Average")</span></h4>
                <p class="text-sm text-muted">A direct comparison showing how the Weighted Score (which factors in vote count/popularity) adjusts the Raw Average Score.</p>
                <div class="p-3 bg-dark rounded shadow-md border" style="height: 350px;">
                    <canvas id="weightedScoreChart"></canvas>
                </div>
            `;
            // 3. Display Top 5 Movies by Weighted Score
            const topMovies = result.top_movies_weighted;
            html += `
            <div class="mt-4">
                <h4 class="text-xl font-semibold mt-4 mb-3">Top 5 Movies by Calculated Weighted Score</h4>
                <p class="text-sm text-muted">Movies that pass the 75th percentile vote threshold (min ${stats.min_votes_for_qualification} votes) are ranked using the IMDB weighted formula for a more robust score.</p>
                <div class="table-responsive">
                    <table class="table table-striped table-hover mt-3 table-dark">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Title</th>
                                <th class="text-center">Year</th>
                                <th class="text-center">Raw Score</th>
                                <th class="text-center">Votes</th>
                                <th class="text-center">Weighted Score</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            topMovies.forEach((movie, index) => {
                html += `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${movie.title}</td>
                        <td class="text-center">${movie.release_year}</td>
                        <td class="text-center">${movie.vote_average.toFixed(2)}</td>
                        <td class="text-center">${formatNumber(movie.vote_count)}</td>
                        <td class="text-center fw-bold">${movie.weighted_score.toFixed(3)}</td>
                    </tr>
                `;
            });
            html += `</tbody></table></div></div>`;

            // 4. Display Top 3 Years by Average Score
            const topYears = result.top_years;
            html += `
            <div id="pandasYearlyAnalysis" class="mt-4">
                <h4 class="text-xl font-semibold mt-4 mb-3">Top 3 Most Recent Years by Average Score</h4>
                <div class="row row-cols-3 g-3 text-center">
            `;
            topYears.forEach((yearData, index) => {
                html += `
                    <div class="col">
                        <div class="p-3 bg-dark rounded shadow-md border-top-3 border-info">
                            <h5 class="text-3xl font-bold text-cyan">${yearData.release_year}</h5>
                            <p class="text-lg">${yearData.vote_average.toFixed(3)} Avg Score</p>
                        </div>
                    </div>
                `;
            });
            html += `</div></div>`;
            pandasContainer.innerHTML = html;

            // RENDER THE CHART - Call the chart function, passing the top movies data from the API response
            renderWeightedScoreChart(result.top_movies_weighted);
        } else {
            statusDiv.className = 'alert alert-danger';
            statusDiv.textContent = `Error: ${result.error || 'Unknown error occurred.'}`;
            pandasContainer.innerHTML = '<p class="text-indigo-500 italic">ERROR ...</p>';
        }
    } catch (error) {
        console.error('Pandas Analysis Error:', error);
        statusDiv.className = 'alert alert-danger';
        statusDiv.textContent = `Connection or network error: ${error.message}`;
        pandasContainer.innerHTML = '<p class="text-indigo-500 italic">ERROR ...</p>';
    } finally {
        // Re-enable button and hide spinner
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-play me-2"></i> Execute Pandas Analysis';
    }
}
// ----------------------------------------------------------------------
// -------------- SCKIT-LEARN HELPERS AND PLOTTER --------- SCI KIT LEARN
// ----- Formats the coefficient object for display. --------------------
// ----- @param {object} coeffs - The coefficients dictionary/object. ---
// ----- @returns {string} HTML string of coefficients. -----------------
// ----------------------------------------------------------------------
function formatCoefficients(coeffs) {
    let html = '';
    for (const [feature, valueString] of Object.entries(coeffs)) {
        html += `
           <div class="flex justify-between items-center py-1 border-b border-gray-700 last:border-b-0">
                <span class="font-mono text-sm text-gray-400">${feature}</span>
                <span class="font-mono text-sm font-semibold text-teal-400">${valueString}</span>
            </div>
        `;
    }
    return html;
}

// ----------------------------------------------------------------------
// -------------- SCKIT-LEARN HELPERS AND PLOTTER --------- SCI KIT LEARN
// --- Renders the CV Scores Bar Chart using Plotly ---------------------
// --- @param {Array<number>} cvScores - Array of cross-validation scores
// ----------------------------------------------------------------------
function renderCVBarChart(cvScores) {
    const cvScorePlotDiv = document.getElementById('cvScorePlot');
    if (!cvScorePlotDiv || !window.Plotly) return;
    
    // Clear previous plot if it exists
    Plotly.purge(cvScorePlotDiv);

    const trace = {
        x: cvScores.map((_, i) => `Fold ${i + 1}`),
        y: cvScores,
        type: 'bar',
        marker: {
            // Use the dark theme color for successful folds (R2 > 0.8)
            color: cvScores.map(score => score > 0.8 ? barWeightedColor : '#f87171'), 
            line: {
                color: darkGridColor,
                width: 0.5
            }
        },
        opacity: 0.8
    };
    const layout = {
        xaxis: { 
            title: 'Cross-Validation Fold', 
            gridcolor: darkGridColor, 
            tickfont: { color: darkTextColor },
            titlefont: { color: darkTextColor }
        },
        yaxis: { 
            title: 'R² Score', 
            range: [-2.5, 0.1],   // [0, 1],   
            gridcolor: darkGridColor, 
            tickfont: { color: darkTextColor },
            titlefont: { color: darkTextColor }
        },
        // Apply dark theme background colors
        paper_bgcolor: darkBgColor, // Outside plot area
        plot_bgcolor: darkBgColor, // Inside plot area
        font: { color: darkTextColor }, // Default font color
        margin: { t: 10, b: 40, l: 40, r: 10 },
        responsive: true
    };

    Plotly.newPlot(cvScorePlotDiv, [trace], layout, { displayModeBar: false, responsive: true });
}

// ----------------------------------------------------------------------
// -------------- SCKIT-LEARN HELPERS AND PLOTTER --------- SCI KIT LEARN
// --- Renders the 3D surface plot for the Weighted Score formula -------
// ----------------------------------------------------------------------
async function render3DPlot(predictionData) {   
    const plotStatus = document.getElementById('plotGenerationStatus');
    const plotDiv = document.getElementById('sklearn3DPlot');

    if (!plotStatus || !plotDiv || !window.Plotly) {
        if (plotStatus) plotStatus.innerHTML = 'Plotly library not loaded or container missing.';
        return;
    }

    // Clear previous plot
    Plotly.purge(plotDiv);

    try {
        const response = await smartFetch('/api/get_sklearn_plot_data');    // await fetch replaced with await smartFetch 
        if (!response.ok) {
            const errorJson = await response.json();
            throw new Error(errorJson.error || `Server responded with status ${response.status}`);
        }
        const data = await response.json();
        if (!data.x || !data.y || !data.z) {
            throw new Error('Missing plot data arrays (x, y, or z) from the server.');
        }

        // Plotly 3D Surface Trace 
        const surfaceTrace = { 
            x: data.x[0],   // V data assigned to x  x: data.x[0], 
            y: data.y.map(r => r[0]), // R data assigned to y
            z: data.z,      // W data assigned to z
            type: 'surface',
            // colorscale: 'Viridis',
            colorscale: 'Portland',
            showscale: false,
            name: 'Formula Surface',       
            hovertemplate: 
                '<b>Vote Count (V):</b> %{x:.0f}<br>' +
                '<b>Raw Score (R):</b> %{y}<br>' +
                '<b>Weighted Score (W):</b> %{z}<extra></extra>' 
        };

        // Start with the surface trace
        let plotTraces = [surfaceTrace]; 
        // let predTrace = null;      

        // 3. Add Prediction Point Trace
        if (predictionData && predictionData.input && predictionData.predicted_weighted_score) {
            const inputStr = predictionData.input;
            
            // 1. Match Raw Score (R): Look for "Raw Score: " followed by a number (integer or decimal)
            const rawScoreMatch = inputStr.match(/Raw Score:\s*(\d+\.?\d*)/);
            // 2. Match Votes (V): Look for "Votes: " followed by an integer
            const votesMatch = inputStr.match(/Votes:\s*(\d+)/);

            console.log(`[3D Plot Debug] Parsing Attempt. Input string: "${inputStr}"`);
            
            if (rawScoreMatch && votesMatch) {
                const rawScoreInput = parseFloat(rawScoreMatch[1]); // R (Y-axis)
                const votesInput = parseInt(votesMatch[1], 10); // V (X-axis)
                const predictedW = parseFloat(predictionData.predicted_weighted_score); // W (Z-axis)

                // --- DEBUGGING LOG (VITAL STEP) ---
                console.log(`[3D Plot Debug] Regex Success. Extracted coordinates: V (X-axis)=${votesInput}, R (Y-axis)=${rawScoreInput}, W (Z-axis)=${predictedW}`); 
                
                predTrace = {
                    x: [votesInput], 
                    y: [rawScoreInput], 
                    z: [predictedW], 
                    mode: 'markers',
                    type: 'scatter3d',
                    marker: {
                        size: 5, // INCREASED SIZE for visibility
                        color: '#fff', 
                        symbol: 'star', // Use a clear symbol
                        line: {
                            color: '#fff', // '#000000' Black border
                            width: 1
                        }
                    },
                    name: 'Model Prediction',
                    hovertemplate: 
                        '<b>MODEL PREDICTION (W)</b><br>' +
                        'Vote Count (V): %{x}<br>' +
                        'Raw Score (R): %{y}<br>' +
                        'Weighted Score (W): %{z:.3f}<extra></extra>' 
                };
                plotTraces.push(predTrace);
            } else {
                 // --- DEBUGGING LOG (VITAL STEP) ---
                 console.error(`[3D Plot Debug] Failed to parse coordinates from input string. Input was: ${inputStr}. Raw Score Match: ${rawScoreMatch}, Votes Match: ${votesMatch}`);
                 const predictionDetailsDiv = document.getElementById('predictionDetails');
                 if (predictionDetailsDiv) {
                     // Optionally show a failure message on the UI
                     predictionDetailsDiv.innerHTML += '<br><span class="text-red-500 font-bold"> (Plot Error: V/R coordinates could not be parsed.)</span>';
                 }
            }
        } else {
                    console.warn("[3D Plot Debug] Skipping prediction point: Prediction data is incomplete or missing.");
        }

        const layout = {
            // Apply dark theme background colors
            paper_bgcolor: darkBgColor,
            plot_bgcolor: darkBgColor,
            font: { color: darkTextColor },
            
            scene: {
                // Change X/Y/Z to V/R/W
                xaxis: { 
                    title: 'Vote Count (V)', 
                    tickformat: ',d',
                    gridcolor: darkGridColor,
                    tickfont: { color: darkTextColor },
                    titlefont: { color: darkTextColor }
                },
                yaxis: { 
                    title: 'Raw Score (R)', 
                    range: [5, 10],
                    gridcolor: darkGridColor,
                    tickfont: { color: darkTextColor },
                    titlefont: { color: darkTextColor }
                },
                zaxis: { 
                    title: 'Weighted Score (W)', 
                    range: [5, 10],
                    gridcolor: darkGridColor,
                    tickfont: { color: darkTextColor },
                    titlefont: { color: darkTextColor }
                },
                aspectmode: 'cube',
                bgcolor: darkBgColor,
            },
            title: `W = (v / (v + ${data.m_constant})) * R + (${data.m_constant} / (v + ${data.m_constant})) * ${data.C_constant}`,
            height: 440,
            margin: { l: 0, r: 0, b: 0, t: 30 }
        };

        // Plot both the surface and the scatter point(s)
        Plotly.newPlot(plotDiv, plotTraces, layout, { responsive: true, displayModeBar: false });
        plotStatus.innerHTML = `Plot generated successfully. (C=${data.C_constant}, m=${data.m_constant})`;

    } catch (error) {
        console.error('3D Plot Error:', error);
        plotStatus.innerHTML = `
            <span class="text-red-700">Error generating 3D plot: ${error.message || 'Plot data failed to load.'}</span>
        `;
    }
}

// ----------------------------------------------------------------------
// -------------- SCKIT-LEARN HELPERS AND PLOTTER --------- SCI KIT LEARN
// --- Renders the results for the linear regression model --------------
// --- Renders the results for the tree regression model ----------------
// --- This is the function that structures the entire output section.---
// ----------------------------------------------------------------------
let lastSklearnResult = null; // ADDED

function renderResults(data) {
    lastSklearnResult = data; // ADDED: store

    // ADDED: Respond to selector
    const selector = document.getElementById('sklearnModelTypeSelector');
    const selectedModel = currentModelType; // CHANGED

    let metrics, modelInfo, coefficients, cvScores, predLabel, examplePred, importances;
    if (selectedModel === 'tree') { // ADDED
        metrics = data.tree_metrics;
        modelInfo = data.tree_model_info || {};
        coefficients = data.tree_feature_importances || {}; // Use importances (not coefficients)
        cvScores = data.tree_cv_scores || [];
        predLabel = "Predicted Weighted Score (Tree):";
        examplePred = {
            input: data.example_prediction?.input,
            predicted_weighted_score: data.example_prediction?.predicted_weighted_score_tree
        };
        importances = coefficients;
    } else {
        metrics = data.linear_metrics;
        modelInfo = data.linear_model_info || {};
        coefficients = data.feature_coefficients || {};
        cvScores = data.lin_cv_scores || [];
        predLabel = "Predicted Weighted Score (Linear):";
        examplePred = {
            input: data.example_prediction?.input,
            predicted_weighted_score: data.example_prediction?.predicted_weighted_score_linear
        };
        importances = null;
    }

    // ...generate metricsHtml as before, but maybe update the headers depending on model...
    // CHANGED: Insert (Linear/Tree) where appropriate, example:
    const metricsHtml = `
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <!-- Coefficients Panel -->
            <div class="p-3 py-0 pt-2 bg-gray-800 rounded-lg shadow-xl border border-dark text-gray-200">
                <p class="ps-2 text-sm font-medium">R² Score (Test): ${metrics.r2_score || 'N/A'}</p>
                <p class="ps-2 text-sm font-medium">Mean Squared Error (MSE): ${metrics.mean_squared_error || 'N/A'}</p>
                <p class="ps-2 text-sm font-medium">CV Mean R² Score: ${metrics.cross_validation_mean_r2 || 'N/A'}</p>
            </div>
            <!-- Cross-Validation Plot Container -->
            <div class="p-4 bg-gray-800 rounded-lg shadow-xl border border-dark">
                <h4 class="text-lg fw-bold mb-3 text-gray-200">5-Fold Cross-Validation Scores</h4>
                <div id="cvScorePlot" class="w-full" style="height: 300px;"></div>
            </div>
        </div>

        <!-- Example Prediction Section -->
        <div class="p-4 py-1 bg-yellow-900 border border-dark rounded-lg shadow-inner text-gray-200">
            <h5 class="fw-bold mt-3">Analytical Conclusion:</h5>
            <p class="small">
                ${selectedModel === 'tree' 
                    ? 'Random Forest is a non-linear model and may capture more complex effects. Compare results with Linear Regression.'
                    : 'The linear regression model gives insight into additive, linear effects of the predictors.'
                }
            </p>
            <hr class="my-3 border-dark">
            <!-- Test Prediction -->
            <h5 class="fw-bold mt-3">Model Test Prediction:</h5>
            <p class="text-sm text-yellow-200" id="predictionDetails">
                Input: <span class="font-mono">${examplePred.input || 'N/A'}</span> <br>
                ${predLabel} <span class="font-bold text-lg">${examplePred.predicted_weighted_score || 'N/A'}</span><br>
                (See white dot on the surface of the 3D graph.)
            </p>
        </div>
        <!-- 3D plot container -->
        <div id="sklearn3DPlotContainer" class="mt-6">
            <h4 class="text-xl font-semibold text-gray-200 my-3">Weighted Score (W) Formula Visualization</h4>
            <p class="text-gray-400 mb-4 text-sm">
                This 3D plot visualizes the core IMDB Weighted Rating formula W = (v / (v + m)) * R + (m / (v + m)) * C, 
                showing how the final score changes based on the Raw Score (R, y-axis) and Vote Count (v, x-axis).
            </p>
            <div id="sklearn3DPlot" class="bg-gray-800 rounded-lg shadow-xl"></div>
            <div id="plotGenerationStatus" class="mt-2 text-sm text-center text-gray-400">Generating 3D surface plot...</div>
        </div>
    `;

    const outputDiv = document.getElementById('sklearnResults');
    if (outputDiv) outputDiv.innerHTML = metricsHtml;

    // Render CV Bar Chart
    renderCVBarChart(cvScores);

    // 3. Render the 3D Plot
    if (window.Plotly) {
        render3DPlot(examplePred);
    } else {
        console.error("[Plotly Check] Plotly is not loaded, skipping 3D plot rendering.");
    }
}
/**
 * Handles the API call to the backend for Scikit-learn analysis.
 */
async function runSklearnAnalysis() {
    const button = document.getElementById('runSklearnAnalysisBtn');
    const statusDiv = document.getElementById('sklearnStatus');
    const outputDiv = document.getElementById('sklearnResults');

    if (!button || !statusDiv || !outputDiv) return;

    // 1. UI State: Disable button and show loading  
    button.disabled = true;
    statusDiv.classList.remove('bg-red-100', 'text-red-800', 'bg-green-100', 'text-green-800');
    statusDiv.classList.add('bg-blue-900', 'text-blue-200'); // Dark theme loading
    outputDiv.innerHTML = '<i class="fas fa-sync fa-spin mr-2"></i> Training Linear Regression Model...';
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> running ... ';
    await sleep(1000);  // 2. Wait for 1000ms (1 second)

    const endpoint = '/api/run_sklearn_analysis';

    try {
        const response = await smartFetch(endpoint, {    // await fetch replaced with await smartFetch 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Pass the features used by the model
            body: JSON.stringify({ features: ['vote_average', 'vote_count', 'popularity'] }) 
        });

        if (!response.ok) {
            let errorText = await response.text();
            try {
                const errorJson = JSON.parse(errorText);
                errorText = errorJson.error || errorText;
            } catch (e) {
                // If parsing fails, use the raw text or default message
            }
            throw new Error(`Server status ${response.status}: ${errorText || 'Unknown error'}`);
        }

        const data = await response.json();

        // 2. Success State
        statusDiv.classList.remove('bg-blue-900', 'text-blue-200');
        statusDiv.classList.add('bg-green-900', 'text-green-200'); 
        statusDiv.innerHTML = '<i class="fas fa-check-circle mr-2"></i> Model Trained and Evaluated Successfully!<br>'

        + 'While the code and workflow complete successfully, the cross-validation performance is consistently poor or even negative for both Linear Regression and Random Forest models.<br>'
        + 'This is a valuable lesson: a negative R² means the model predicts worse than simply outputting the average score for every film!.<br>'
        + '<br>'
        + '<b>Why?</b> With only 120 movies, the data is limited and very skewed. Additionally, several features (such as popularity and vote count) are highly correlated, leading to issues with multicollinearity.<br>'
        + 'The target (weighted score) is also a non-linear function of the features, making it difficult for these models to find robust patterns, especially with so little data.<br>'
        + '<br>'
        + '<b>Takeaway:</b> This catastrophic result is not a bug in the code, but a real reflection of data and feature limitations.<br>'
        + 'With a larger, more diverse dataset (or by engineering more independent features), model performance could improve. Demonstrating and analyzing failure cases like this is as important as showing successes—it is how real data science works.<br>'
        ;
        renderResults(data);

    } catch (error) {
        console.error('Scikit-learn Analysis Error:', error);
        // 3. Error State
        statusDiv.classList.remove('bg-blue-900', 'text-blue-200', 'bg-green-900', 'text-green-200');
        statusDiv.classList.add('bg-red-900', 'text-red-200'); // Dark theme error
        statusDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle mr-2"></i> <strong>Error:</strong> ${error.message || 'Could not connect to the backend ML service.'}
        `;
        outputDiv.innerHTML = `
            <div class="p-4 text-sm text-gray-400 bg-gray-900 rounded-lg">
                Check the console for details. The server endpoint <code>${endpoint}</code> might be missing or failed to process the request.
            </div>
        `;
    } finally {
        // 4. Re-enable button and hide loading 
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-play me-2"></i> Run Predictive Model';
    }
}

// ----------------------------------------------------------------------
// -------------------- NUMPY HELPERS ----------------------------- NUMPY
// --- Executes the NumPy analysis endpoint and displays ----------------
// --- the standardized and normalized results in a tabular format ------
// ---------------------------------------------------------------------- 
async function runNumpyAnalysis() {
    const button = document.getElementById('runNumpyAnalysisBtn');
    const resultsDiv = document.getElementById('numpyResults');
    if (!button || !resultsDiv) return;
    button.disabled = true;                               // UI State: Disable button and show loading
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> running ... ';
    resultsDiv.innerHTML = '';
    resultsDiv.classList.add('hidden');
    await sleep(1000);  // 2. Wait for 1000ms (1 second)
    try {
        const response = await smartFetch('/api/run_numpy_analysis');    // await fetch replaced with await smartFetch 
        if (!response.ok) {
            const errorJson = await response.json();
            throw new Error(errorJson.error || `Server responded with status ${response.status}`);
        }
        const data = await response.json();

        // Build the HTML table and stats block
        let html = `
            <h3 class="text-xl font-bold mb-3 text-white">NumPy Preprocessing Demo (Top ${data.results.length} Movies)</h3>
            <p class="text-sm mb-4 text-gray-300">
                Data is transformed using pure NumPy: Z-Score Standardization (Mean: ${data.stats.mean}, StdDev: ${data.stats.std}) and 
                Min-Max Normalization (Range: ${data.stats.min} to ${data.stats.max}).
            </p>
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-600 bg-gray-800 rounded-lg shadow-lg">
                    <thead class="bg-gray-700">
                        <tr>
                            <th class="px-3 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Title</th>
                            <th class="px-3 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Raw Score (0-10)</th>
                            <th class="px-3 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Z-Score (Standardized)</th>
                            <th class="px-3 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Min-Max (Normalized)</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-700">
        `;

        data.results.forEach(movie => {
            html += `
                <tr class="hover:bg-gray-700">
                    <td class="px-3 py-2 whitespace-nowrap text-sm font-medium text-white">${movie.title}</td>
                    <td class="px-3 py-2 whitespace-nowrap text-sm text-yellow-400">${movie.raw}</td>
                    <td class="px-3 py-2 whitespace-nowrap text-sm text-red-300">${movie.z_score}</td>
                    <td class="px-3 py-2 whitespace-nowrap text-sm text-green-300">${movie.min_max}</td>
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;
        resultsDiv.innerHTML = html;
        resultsDiv.classList.remove('hidden');
    } catch (error) {
        console.error("NumPy Analysis Error:", error);
        resultsDiv.innerHTML = `<p class="text-red-400">Error performing NumPy analysis: ${error.message}</p>`;
        resultsDiv.classList.remove('hidden');
    } finally {
        button.disabled = false;                              // Re-enable button and hide loading
        button.innerHTML = '<i class="fas fa-play me-2"></i> Execute numpy Analysis';
    }
}

// ---------------------------------------------------------------------
// ---       Python / C integration Test  /api/runCbridge           ----
// ---------------------------------------------------------------------
async function runCbridge() {
    const outp1 = document.getElementById('CbridgeOutput1');
    const outp2 = document.getElementById('CbridgeOutput2');
    const btn = document.getElementById('CbridgeBtn');
    const Cbridge_ENDPOINT = '/api/runCbridge';
    
    if (!outp1 || !outp2 || !btn) return;
    outp1.innerHTML = `Running C / Python Integration and Performance Benchmark  ...`;
    outp1.classList.remove('text-muted');
    outp2.classList.add('d-none');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> running ... ';
    try {
        const response = await smartFetch(Cbridge_ENDPOINT, {    // await fetch replaced with await smartFetch 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        if (!response.ok) {
            const errorData = await response.json();
            outp1.innerHTML = `Error: ${errorData.error}`;
            return;
        }
        const data = await response.json();
        // 2. Ensure data exists and is not zero to avoid division by zero
        if (!data || !data.time_c || data.time_py === 0) {
            console.error("Invalid benchmark data received");
            outp1.innerHTML = `Error: Invalid benchmark data received`;
            return;
        }
        if (data.result) {
            await sleep(500); 
            const CbridgeCvalue = document.getElementById('CbridgeCvalue');
            const CbridgePvalue = document.getElementById('CbridgePvalue');
            const CbridgeCpercent = document.getElementById('CbridgeCpercent');
            outp1.innerHTML = `Performance Benchmark - TERMINAL OUTPUT:\n${data.result}`
            outp1.classList.add('hidden');
            outp1.style.height = "450px";
            outp1.setAttribute("style", "white-space: pre-wrap; font-family: monospace;");
            outp2.classList.remove('d-none');
            CbridgeCvalue.innerHTML = `~ ${(data.time_c*1000).toFixed(2)} ms`;
            CbridgePvalue.innerHTML = `~ ${(data.time_py*1000).toFixed(2)} ms`;
            CbridgeCpercent.style.width = `${(data.time_c / data.time_py) * 100}%`;
        } else {
             outp1.innerHTML = `Error: endpoint /api/runCbridge returned no output.`;
        }
    } catch (error) {
        outp1.innerHTML = `Network Error: ${error.message}`;
        console.error('Fetch error:', error);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play me-2"></i><span class="text-uppercase">Run</span> Performance Benchmark';
    }
}

// ----------------------------------------------------------------------
// -------------------- EVENT LISTENERS ---------------------------------
// ----------------------------------------------------------------------
window.addEventListener('load', () => {
    // // 0. disable to return to the last position you were viewing
    // if (window.location.hash) {
    //     window.history.replaceState(null, '', window.location.pathname + window.location.search);
    //     // Force scroll to the very top (or to a specific element like the header)
    //     window.scrollTo(0, 0); 
    // }

    // 1. Initialize Scrollspy after all content is loaded
    initializeScrollspyObserver();
    
    // 2. Attach event listener for the Fibonacci Test Button
    const fibBtn = document.getElementById('runFibonacciTestBtn');
    if (fibBtn) {
        fibBtn.addEventListener('click', runFibonacciTest);
    }

    // 3. Attach event listener for the Fibonacci Test Button
    const fibBtnPytest = document.getElementById('runFibonacciPytestBtn');
    if (fibBtnPytest) {
        fibBtnPytest.addEventListener('click', runFibonacciPytest);
    }

    // 4. Attach fetchMovies to the button
    const fetchBtn = document.getElementById('fetchMoviesBtn');
    if (fetchBtn) {
        fetchBtn.addEventListener('click', fetchMovies);
    }

    // 5. Attach runPandasAnalysis to the new Pandas button
    const pandasBtn = document.getElementById('runPandasAnalysisBtn');
    if (pandasBtn) {
        pandasBtn.addEventListener('click', runPandasAnalysis);
    }

    // 6. Attach runSklearnAnalysis to the new Scikit-learn button
    const sklearnBtn = document.getElementById('runSklearnAnalysisBtn');
    if (sklearnBtn) {
        sklearnBtn.addEventListener('click', runSklearnAnalysis);
    }

    // 7. Attach runNumpyAnalysis to the new NumPy button
    const numpyBtn = document.getElementById('runNumpyAnalysisBtn');
    if (numpyBtn) {
        numpyBtn.addEventListener('click', runNumpyAnalysis);
    }

    // Attach event listener for the C/Python integration Test Button
    const CbridgeBtn = document.getElementById('CbridgeBtn');
    if (CbridgeBtn) {
        CbridgeBtn.addEventListener('click', runCbridge);
    }
});

// ----------------------------------------------------------------------
// -------------- SCKIT-LEARN HELPERS AND PLOTTER --------- SCI KIT LEARN
// --- Store current model (default to 'linear') ------------------------
// ----------------------------------------------------------------------
let currentModelType = 'linear';

document.addEventListener('DOMContentLoaded', function() {
    // Dropdown handler for model choice
    document.querySelectorAll('.dropdown-menu .dropdown-item').forEach(function(item) {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const modelType = this.getAttribute('data-model');
            currentModelType = modelType;
            // Update button label
            document.getElementById('selectedModelText').textContent = this.textContent.trim();
            if (lastSklearnResult) renderResults(lastSklearnResult); // rerender, see previous code
        });
    });
    // Kick off a rerender if needed (optional see comments)
    if (lastSklearnResult) renderResults(lastSklearnResult);
});


// ----------------------------------------------------------------------
// -------------------- Smart Fetch Wrapper -----------------------------
//  A wrapper for fetch that shows a "Waking Up" overlay if the request -
// takes longer than 1000ms (typical of a Cloud Run cold start).        -
// ----------------------------------------------------------------------
async function smartFetch(url, options = {}) {
    const overlay = document.getElementById('backend-wakeup-overlay');
    
    // 1. Set a timer to show the overlay if the request takes > 1000ms
    const wakeupTimer = setTimeout(() => {
        if (overlay) {
            overlay.classList.remove('d-none');
            overlay.style.setProperty('display', 'flex', 'important');
        }
    }, 1500);

    try {
        // 2. Perform the ACTUAL request
        const response = await fetch(url, options);
        return response;
    } catch (error) {
        console.error("SmartFetch Error:", error);
        throw error; // Pass error to the calling function (e.g., TensorFlow script)
    } finally {
        // 3. CRITICAL: Always clear the timer and hide the overlay
        clearTimeout(wakeupTimer);
        if (overlay) {
            overlay.classList.add('d-none');
            overlay.style.setProperty('display', 'none', 'important');
        }
    }
}

// ----------------------------------------------------------------------
// -------------------- Smart Fetch Wrapper ----------- TEST MODE VERSION
//  A wrapper for fetch that shows a "Waking Up" overlay if the request -
//  This version skips the actual fetch and keeps the overlay visible   -
//  for 10 seconds so you can test the UI design.                       -
// ----------------------------------------------------------------------
async function smartFetchX(url, options = {}) {
    const overlay = document.getElementById('backend-wakeup-overlay');
    
    console.log("SmartFetch: Test Mode Activated for", url);

    // 1. Show the overlay after 1 second (simulating a slow start)
    const wakeupTimer = setTimeout(() => {
        if (overlay) {
            console.log("SmartFetch: Container took too long, showing UI...");
            overlay.classList.remove('d-none');
            overlay.style.setProperty('display', 'flex', 'important');
        }
    }, 1000);

    // 2. Simulate a 10-second wait instead of doing the real fetch
    return new Promise((resolve) => {
        setTimeout(() => {
            console.log("SmartFetch: 10s test finished. (Normally would hide now)");
            
            // To test the "hiding" logic, uncomment the lines below:

            clearTimeout(wakeupTimer);
            if (overlay) {
                overlay.classList.add('d-none');
                overlay.style.setProperty('display', 'none', 'important');
            }
            resolve({ ok: true, json: () => ({ result: "Test Finished" }) });

        }, 10000);
    });
}

// ----------------------------------------------------------------------
// ----------------------- Heartbeat script -----------------------------
//  for better User Experience:                                       ---
//  script pings the server every 4 minutes while the tab is open.    ---
//  Keep the container warm while the user is active on the page      ---
// ----------------------------------------------------------------------
// setInterval(() => {
//     fetch('/api/health').catch(() => {}); // Minimal ping
// }, 240000); // 4 minutes