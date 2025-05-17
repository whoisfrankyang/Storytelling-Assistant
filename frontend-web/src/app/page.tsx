'use client';

import { useState } from 'react';
import FileUpload from '../components/FileUpload';
import ResultsDisplay from '../components/ResultsDisplay';

export default function Home() {
  const [results, setResults] = useState<string>('');
  const [inputText, setInputText] = useState<string>('');
  const [mode, setMode] = useState<string>('general');

  const handleFileUpload = async (file: File) => {
    // Here you can implement the file processing logic
    // For now, we'll just read the file content
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('mode', mode);
  
      const response = await fetch('http://localhost:8000/process', {
        method: 'POST',
        body: formData,
      });
  
      if (!response.ok) {
        throw new Error('Failed to process file');
      }
  
      const data = await response.json();
      setResults(data.result);
    } catch (error) {
      console.error('Error processing file:', error);
      setResults('Error processing file. Please try again.');
    }
  };

  const handleTextSubmit = async () => {
    try {
      const response = await fetch('http://localhost:8000/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_data: inputText, mode }),
      });

      if (!response.ok) {
        throw new Error('Failed to process text');
      }

      const data = await response.json();
      setResults(data.result);
    } catch (error) {
      console.error('Error processing text:', error);
      setResults('Error processing text. Please try again.');
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">
          Storytelling Assistant
        </h1>
        {/* Dropdown for version selection */}
        <div className="flex justify-center mb-6">
          <label className="mr-2 font-semibold text-gray-700" htmlFor="mode-select">Output Version:</label>
          <select
            id="mode-select"
            className="border border-gray-300 rounded px-3 py-1"
            value={mode}
            onChange={e => setMode(e.target.value)}
          >
            <option value="general">General</option>
            <option value="investor">Investor</option>
            <option value="conference">Conference</option>
          </select>
        </div>
        {/* Top half: two columns */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          {/* Left: File Upload */}
          <div className="bg-white rounded-lg shadow-lg p-6 flex flex-col justify-start">
            <FileUpload onFileUpload={handleFileUpload} />
          </div>
          {/* Right: Text Input */}
          <div className="bg-white rounded-lg shadow-lg p-6 flex flex-col justify-start">
            <h2 className="text-2xl font-semibold mb-4 text-gray-700">Enter Text</h2>
            <textarea
              className="w-full h-40 p-3 border border-gray-300 rounded mb-4 resize-none focus:outline-none focus:ring-2 focus:ring-blue-200"
              placeholder="Type or paste your text here..."
              value={inputText}
              onChange={e => setInputText(e.target.value)}
            />
            <button
              className="self-end px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
              onClick={handleTextSubmit}
            >
              Submit
            </button>
          </div>
        </div>
        {/* Bottom: Results */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <ResultsDisplay results={results} />
        </div>
      </div>
    </main>
  );
}

