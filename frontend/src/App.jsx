import React, { useState } from 'react';
import SegmentationViewer from './components/SegmentationViewer';
import MetricsDashboard from './components/MetricsDashboard';

function App() {
  const [activeTab, setActiveTab] = useState('segment'); // 'segment' or 'metrics'

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans selection:bg-purple-500/30">
      {/* Header */}
      <header className="pt-12 pb-8 px-4 border-b border-white/10 bg-black/40 backdrop-blur-md sticky top-0 z-10 shadow-lg">
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-500 mb-3 drop-shadow-sm">
            U-Net Segmentation
          </h1>
          <p className="text-lg md:text-xl text-gray-400 font-medium">
            Oxford-IIIT Pet Dataset &middot; Semantic Segmentation
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-10">
        
        {/* Tabs */}
        <div className="flex justify-center space-x-4 mb-10">
          <button
            onClick={() => setActiveTab('segment')}
            className={`px-8 py-3 rounded-full text-lg font-semibold transition-all duration-300 relative overflow-hidden group ${
              activeTab === 'segment' 
                ? 'text-white bg-white/10 shadow-[0_0_20px_rgba(168,85,247,0.4)]' 
                : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
            }`}
          >
            <span className="relative z-10">Segment Image</span>
            {activeTab === 'segment' && (
              <div className="absolute inset-0 border-2 border-transparent rounded-full [background:linear-gradient(to_right,#3b82f6,#a855f7)_border-box] [mask-image:linear-gradient(white,white),linear-gradient(white,white)] [mask-origin:border-box] [mask-clip:padding-box,border-box] [mask-composite:exclude]" />
            )}
          </button>

          <button
            onClick={() => setActiveTab('metrics')}
            className={`px-8 py-3 rounded-full text-lg font-semibold transition-all duration-300 relative overflow-hidden group ${
              activeTab === 'metrics' 
                ? 'text-white bg-white/10 shadow-[0_0_20px_rgba(168,85,247,0.4)]' 
                : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
            }`}
          >
            <span className="relative z-10">Model Metrics</span>
            {activeTab === 'metrics' && (
              <div className="absolute inset-0 border-2 border-transparent rounded-full [background:linear-gradient(to_right,#3b82f6,#a855f7)_border-box] [mask-image:linear-gradient(white,white),linear-gradient(white,white)] [mask-origin:border-box] [mask-clip:padding-box,border-box] [mask-composite:exclude]" />
            )}
          </button>
        </div>

        {/* Tab Content */}
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
          {activeTab === 'segment' ? <SegmentationViewer /> : <MetricsDashboard />}
        </div>
        
      </main>
      
      {/* Footer */}
      <footer className="text-center py-6 text-gray-500 border-t border-white/5">
        <p>Built by neeraj214 &copy; {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;
