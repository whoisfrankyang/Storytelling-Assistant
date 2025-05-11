interface ResultsDisplayProps {
  results: string;
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
  return (
    <div className="w-full">
      <h2 className="text-2xl font-semibold mb-4 text-gray-700">Results</h2>
      
      <div className="bg-gray-50 rounded-lg p-6 min-h-[300px]">
        {results ? (
          <pre className="whitespace-pre-wrap text-gray-800 font-mono text-sm">
            {results}
          </pre>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            Upload a file to see results
          </div>
        )}
      </div>
    </div>
  );
}