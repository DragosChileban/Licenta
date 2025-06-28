// export function setupCounter(element) {
//   let counter = 0
//   const setCounter = (count) => {
//     counter = count
//     element.innerHTML = `count is ${counter}`
//   }
//   element.addEventListener('click', () => setCounter(counter + 1))
//   setCounter(0)
// }
// // main.js

import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
    const container = document.getElementById('viewerContainer');
    


    const viewer = new GaussianSplats3D.Viewer({
      domElement: container,
      cameraUp: [0, -1, -0.17],
      initialCameraPosition: [-5, -1, -1],
      initialCameraLookAt: [-1.72477, 0.05395, -0.00147],
      sphericalHarmonicsDegree: 2
    });

    const canvas = viewer.renderer.domElement;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';      // removes unwanted space under canvas
    canvas.style.objectFit = 'cover';    // optional: ensures it stretches fully
    canvas.style.position = 'absolute';  // aligns to the container if needed
    canvas.style.top = '0';
    canvas.style.left = '0';

    container.appendChild(canvas);

    const path = '/Users/dragos/Licenta/data/testapp/demo/3dgs.ply';

    viewer.addSplatScene(path, {
      progressiveLoad: true
    }).then(() => {
      viewer.start();
    });