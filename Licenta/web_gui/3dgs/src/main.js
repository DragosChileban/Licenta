const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    let selectedFile = null;

    // Drag events
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length) {
        selectedFile = files[0];
        dropZone.querySelector('p').textContent = `Selected: ${selectedFile.name}`;
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
        selectedFile = fileInput.files[0];
        dropZone.querySelector('p').textContent = `Selected: ${selectedFile.name}`;
        }
    });

// async function sendFile() {
//         if (!selectedFile) {
//         alert('Please select a file first.');
//         return;
//         }

//         const formData = new FormData();
//         formData.append('file', selectedFile);

//         try {
//         const response = await fetch('http://localhost:5100/upload', {
//             method: 'POST',
//             body: formData
//         });

//         const result = await response.json();

//         const responseDiv = document.getElementById('response');
//         responseDiv.style.display = 'block';
//         responseDiv.innerHTML = `<strong>Response:</strong> ${result.message}`;
//         } catch (error) {
//         console.error('Upload failed:', error);
//         alert('Failed to upload the file.');
//         }
//     }


async function sendFile() {
    if (!selectedFile) {
        alert('Please select a file first.');
        return;
    }

    // Retrieve the selected quality
    const quality = document.getElementById('qualitySelect').value;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('quality', quality);

    const loadingDiv = document.getElementById('loading');
    const responseDiv = document.getElementById('response');

    try {
        // Show the loading spinner
        loadingDiv.style.display = 'block';
        responseDiv.style.display = 'none';

        const response = await fetch('http://localhost:5100/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        // Display the response and hide spinner
        responseDiv.innerHTML = `<strong>Response:</strong> ${result.message}`;
        responseDiv.style.display = 'block';
    } catch (error) {
        console.error('Upload failed!', error);
        alert('Failed to upload the file.');
    } finally {
        // Always hide the loading spinner
        loadingDiv.style.display = 'none';
    }
}

