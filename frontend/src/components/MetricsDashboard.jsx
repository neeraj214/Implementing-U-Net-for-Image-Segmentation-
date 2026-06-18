import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { getMetrics } from '../api/unetApi';

const COLORS = {
  Pet: '#ef4444',
  Background: '#22c55e',
  Border: '#3b82f6',
};

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await getMetrics();
        setMetrics(data);
      } catch (err) {
        setError(err.message || 'Failed to load metrics');
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64 w-full">
        <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full max-w-4xl mx-auto p-6 bg-red-500/10 border border-red-500/30 rounded-xl text-center text-red-200">
        <p>{error}</p>
      </div>
    );
  }

  const evalData = metrics?.eval_results || {};
  const trainData = metrics?.train_meta || {};
  
  const testAcc = evalData.test_accuracy || 0;
  const meanIoU = evalData.mean_iou || 0;
  const epochs = trainData.epochs_trained || 0;
  
  let trainTime = '0m 0s';
  if (trainData.training_time) {
    const mins = Math.floor(trainData.training_time / 60);
    const secs = Math.floor(trainData.training_time % 60);
    trainTime = `${mins}m ${secs}s`;
  }
  
  const perClassIou = evalData.per_class_iou || {};
  const iouData = [
    { name: 'Pet', iou: perClassIou.Pet || 0, color: COLORS.Pet },
    { name: 'Background', iou: perClassIou.Background || 0, color: COLORS.Background },
    { name: 'Border', iou: perClassIou.Border || 0, color: COLORS.Border },
  ];

  return (
    <div className="w-full max-w-5xl mx-auto space-y-8 text-gray-100">
      <h2 className="text-3xl font-bold text-center mb-8 drop-shadow-md">Model Performance</h2>
      
      {/* Section 1: Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard title="Test Accuracy" value={`${(testAcc * 100).toFixed(2)}%`} icon="🎯" />
        <MetricCard title="Mean IoU" value={`${(meanIoU * 100).toFixed(2)}%`} icon="📊" />
        <MetricCard title="Epochs Trained" value={epochs} icon="🔄" />
        <MetricCard title="Training Time" value={trainTime} icon="⏱️" />
      </div>

      {/* Section 2: Per-class IoU Chart */}
      <div className="bg-white/10 backdrop-blur-md border border-white/20 p-6 rounded-2xl shadow-xl">
        <h3 className="text-xl font-semibold mb-6">Per-Class Intersection over Union (IoU)</h3>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={iouData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="name" stroke="rgba(255,255,255,0.7)" />
              <YAxis stroke="rgba(255,255,255,0.7)" domain={[0, 1]} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px' }}
                formatter={(value) => `${(value * 100).toFixed(1)}%`}
              />
              <Bar dataKey="iou" radius={[4, 4, 0, 0]}>
                {iouData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Section 3: Architecture Diagram */}
      <div className="bg-white/5 backdrop-blur-md border border-white/10 p-8 rounded-2xl shadow-xl">
        <h3 className="text-xl font-semibold mb-8 text-center">U-Net Architecture</h3>
        
        <div className="flex flex-col items-center justify-center font-mono text-sm sm:text-base relative w-full overflow-x-auto pb-4">
          
          <div className="flex flex-col md:flex-row items-center w-full min-w-max justify-between px-4">
            
            {/* Encoder */}
            <div className="flex flex-col space-y-4 items-center">
              <Block label="Input Image" size="bg-gray-700" />
              <div className="flex flex-col items-center">
                <span className="text-gray-400 my-1">↓ Conv + MaxPool</span>
                <Block label="Encoder 1" size="bg-blue-900/60 border-blue-500" />
              </div>
              <div className="flex flex-col items-center">
                <span className="text-gray-400 my-1">↓ Conv + MaxPool</span>
                <Block label="Encoder 2" size="bg-blue-800/60 border-blue-400" />
              </div>
              <div className="flex flex-col items-center">
                <span className="text-gray-400 my-1">↓ Conv + MaxPool</span>
                <Block label="Encoder 3" size="bg-blue-700/60 border-blue-300" />
              </div>
            </div>

            {/* Skip Connections */}
            <div className="hidden md:flex flex-col justify-between h-[300px] flex-grow px-4 mt-16">
              <SkipArrow label="Skip Connection 1" />
              <SkipArrow label="Skip Connection 2" />
              <SkipArrow label="Skip Connection 3" />
            </div>

            {/* Decoder */}
            <div className="flex flex-col space-y-4 items-center mt-8 md:mt-0 justify-end h-full">
              <div className="flex flex-col items-center">
                <Block label="Decoder 3" size="bg-emerald-700/60 border-emerald-300" />
                <span className="text-gray-400 my-1">↑ UpConv + Concat</span>
              </div>
              <div className="flex flex-col items-center">
                <Block label="Decoder 2" size="bg-emerald-800/60 border-emerald-400" />
                <span className="text-gray-400 my-1">↑ UpConv + Concat</span>
              </div>
              <div className="flex flex-col items-center">
                <Block label="Decoder 1" size="bg-emerald-900/60 border-emerald-500" />
                <span className="text-gray-400 my-1">↑ UpConv</span>
              </div>
              <Block label="Output Mask" size="bg-gray-700 border-gray-400" />
            </div>
          </div>
          
          {/* Bottleneck */}
          <div className="mt-8 mb-4 border border-purple-500/50 bg-purple-900/30 px-12 py-3 rounded-lg shadow-[0_0_15px_rgba(168,85,247,0.3)]">
            Bottleneck
          </div>

        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, icon }) {
  return (
    <div className="bg-white/10 backdrop-blur-md border border-white/20 p-5 rounded-xl shadow-lg flex items-center space-x-4 hover:bg-white/15 transition-colors">
      <div className="text-3xl">{icon}</div>
      <div>
        <p className="text-sm text-gray-300 font-medium">{title}</p>
        <p className="text-2xl font-bold text-white">{value}</p>
      </div>
    </div>
  );
}

function Block({ label, size }) {
  return (
    <div className={`px-4 py-2 border rounded-md shadow-md text-center ${size} w-32 backdrop-blur-sm`}>
      {label}
    </div>
  );
}

function SkipArrow({ label }) {
  return (
    <div className="relative w-full flex items-center justify-center opacity-60 hover:opacity-100 transition-opacity">
      <div className="w-full h-px bg-gradient-to-r from-blue-500 via-gray-400 to-emerald-500"></div>
      <span className="absolute bg-black/50 px-2 rounded text-xs text-gray-300">{label} →</span>
    </div>
  );
}
