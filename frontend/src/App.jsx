import React, { useState } from 'react';
import SegmentationViewer from './components/SegmentationViewer';
import MetricsDashboard from './components/MetricsDashboard';

function App() {
  const [activeTab, setActiveTab] = useState('segment');

  return (
    <div className="min-h-screen bg-slate-900 text-white py-12 px-4 sm:px-6 lg:px-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-12">
        
        {/* Header */}
        <header className="text-center space-y-4">
          <h1 className="text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-500 drop-shadow-sm">
            U-Net Segmentation
          </h1>
          <p className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto font-medium tracking-wide">
            Oxford-IIIT Pet Dataset &middot; Semantic Segmentation
          </p>
        </header>

        <main>
          {/* Tab Navigation */}
          <div className="flex justify-center mb-12">
            <div className="flex space-x-2 bg-slate-800/60 p-2 rounded-2xl border border-slate-700/50 shadow-xl backdrop-blur-md">
              
              <button 
                onClick={() => setActiveTab('segment')} 
                className="relative group rounded-xl"
              >
                {/* Gradient Border Background */}
                <div className={`absolute inset-0 rounded-xl bg-gradient-to-r from-purple-500 to-blue-500 transition-opacity duration-300 ${activeTab === 'segment' ? 'opacity-100' : 'opacity-0 group-hover:opacity-50'}`}></div>
                {/* Inner button surface */}
                <div className={`relative px-8 py-3 rounded-xl font-bold tracking-wide transition-colors duration-300 m-[2px] ${
                  activeTab === 'segment' 
                  ? 'bg-slate-900 text-white' 
                  : 'bg-slate-800/80 text-slate-400'
                }`}>
                  Segment Image
                </div>
              </button>

              <button 
                onClick={() => setActiveTab('metrics')} 
                className="relative group rounded-xl"
              >
                {/* Gradient Border Background */}
                <div className={`absolute inset-0 rounded-xl bg-gradient-to-r from-purple-500 to-blue-500 transition-opacity duration-300 ${activeTab === 'metrics' ? 'opacity-100' : 'opacity-0 group-hover:opacity-50'}`}></div>
                {/* Inner button surface */}
                <div className={`relative px-8 py-3 rounded-xl font-bold tracking-wide transition-colors duration-300 m-[2px] ${
                  activeTab === 'metrics' 
                  ? 'bg-slate-900 text-white' 
                  : 'bg-slate-800/80 text-slate-400'
                }`}>
                  Model Metrics
                </div>
              </button>

            </div>
          </div>

          {/* Tab Content Area */}
          <div className="mt-8 transition-all duration-500 ease-in-out">
            {activeTab === 'segment' && <SegmentationViewer />}
            {activeTab === 'metrics' && <MetricsDashboard />}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
