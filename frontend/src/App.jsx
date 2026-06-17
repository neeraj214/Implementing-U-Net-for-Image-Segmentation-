import React from 'react';
import SegmentationViewer from './components/SegmentationViewer';

function App() {
  return (
    <div className="min-h-screen bg-slate-900 text-white py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-12">
        <header className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
            U-Net Image Segmentation
          </h1>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Upload an image to segment Pet, Background, and Border regions using our deep learning model.
          </p>
        </header>

        <main>
          <SegmentationViewer />
        </main>
      </div>
    </div>
  );
}

export default App;
