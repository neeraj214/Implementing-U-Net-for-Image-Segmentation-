import React, { useState, useRef } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { segmentImage } from '../api/unetApi';

const COLORS = {
  Pet: '#ef4444', // red
  Background: '#22c55e', // green
  Border: '#3b82f6', // blue
};

export default function SegmentationViewer() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  
  const fileInputRef = useRef(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = async (file) => {
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await segmentImage(file);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to segment image');
    } finally {
      setLoading(false);
    }
  };

  const formatChartData = (distribution) => {
    if (!distribution) return [];
    return [
      { name: 'Pet', value: distribution.pet || 0, color: COLORS.Pet },
      { name: 'Background', value: distribution.background || 0, color: COLORS.Background },
      { name: 'Border', value: distribution.border || 0, color: COLORS.Border },
    ].filter(d => d.value > 0);
  };

  const getDominantClass = (distribution) => {
    if (!distribution) return '';
    let max = -1;
    let dominant = '';
    Object.entries(distribution).forEach(([key, val]) => {
      if (val > max) {
        max = val;
        dominant = key;
      }
    });
    return dominant.charAt(0).toUpperCase() + dominant.slice(1);
  };

  const chartData = result ? formatChartData(result.class_distribution) : [];
  const dominantClass = result ? getDominantClass(result.class_distribution) : '';

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl shadow-2xl text-white">
      <h2 className="text-2xl font-bold mb-6 text-center text-gray-100">U-Net Image Segmentation</h2>
      
      {!imagePreview && (
        <div 
          className="border-2 border-dashed border-gray-400/50 rounded-xl p-12 text-center cursor-pointer hover:bg-white/5 transition-colors"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept="image/*" 
            className="hidden" 
          />
          <div className="text-gray-300">
            <svg className="mx-auto h-12 w-12 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
              <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <p className="text-lg">Drag & drop an image here, or click to upload</p>
          </div>
        </div>
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center py-12">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
          <p className="text-lg text-blue-200">Segmenting...</p>
        </div>
      )}

      {error && (
        <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200 text-center">
          {error}
          <button 
            onClick={() => { setError(null); setImagePreview(null); }}
            className="block mx-auto mt-2 text-sm underline"
          >
            Try Again
          </button>
        </div>
      )}

      {result && !loading && (
        <div className="mt-8 space-y-8 animate-in fade-in duration-500">
          <div className="flex flex-col md:flex-row justify-center items-center gap-8">
            <div className="flex flex-col items-center">
              <h3 className="text-lg font-medium mb-3 text-gray-200">Original Image</h3>
              <img src={imagePreview} alt="Original" className="w-64 h-64 object-cover rounded-lg shadow-lg border border-white/10" />
            </div>
            
            <div className="flex flex-col items-center">
              <h3 className="text-lg font-medium mb-3 text-gray-200">Segmentation Mask</h3>
              <img src={`data:image/png;base64,${result.mask_base64}`} alt="Mask" className="w-64 h-64 object-cover rounded-lg shadow-lg border border-white/10" />
            </div>
          </div>

          {chartData.length > 0 && (
            <div className="bg-black/20 p-6 rounded-xl border border-white/10">
              <h3 className="text-xl font-medium mb-4 text-center">Class Distribution</h3>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      formatter={(value) => `${value.toFixed(2)}%`}
                      contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px' }}
                    />
                    <Legend 
                      verticalAlign="bottom" 
                      height={36}
                      formatter={(value, entry) => <span className="text-gray-200">{value}</span>}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              
              <div className="mt-6 flex justify-center">
                <div className="px-6 py-2 bg-gradient-to-r from-indigo-500/30 to-purple-500/30 border border-indigo-500/50 rounded-full">
                  <span className="text-gray-300 mr-2">Dominant Class:</span>
                  <span className="font-bold text-white">{dominantClass}</span>
                </div>
              </div>
            </div>
          )}
          
          <div className="flex justify-center pt-4">
            <button 
              onClick={() => { setResult(null); setImagePreview(null); }}
              className="px-6 py-2 bg-white/10 hover:bg-white/20 transition-colors border border-white/30 rounded-lg text-white"
            >
              Upload Another Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
