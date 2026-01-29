import { Home, Search, Heart } from 'lucide-react';
import { useState, useEffect } from 'react';
import axios from 'axios';

const Sidebar = ({ view, setView, onRetrain, onHome }) => {
  const [modelVersion, setModelVersion] = useState("Loading...");

  useEffect(() => {
      // Fetch initial version
      fetchVersion();
      
      // Poll for version updates (simple way to reflect retraining changes)
      const interval = setInterval(fetchVersion, 10000);
      return () => clearInterval(interval);
  }, []);

  const fetchVersion = async () => {
      try {
          const res = await axios.get('http://localhost:8000/model-info');
          setModelVersion(res.data.version);
      } catch (err) {
          console.error("Version fetch error", err);
      }
  };

  return (
    <div className="w-64 bg-black h-full flex flex-col gap-2 pt-6 px-2 pb-2">
    <div className="px-4 mb-6">
        <h1 className="text-white text-2xl font-bold flex items-center gap-2">
          <span className="text-3xl">🎵</span> Spotify MLOps
        </h1>
    </div>
      
      <div className="bg-[#121212] rounded-lg p-4 flex flex-col gap-4">
        <div 
            onClick={onHome}
            className={`flex items-center gap-4 font-bold cursor-pointer transition-colors ${view === 'home' ? 'text-white' : 'text-[#b3b3b3] hover:text-white'}`}
        >
            <Home size={24} />
            <span>Home</span>
        </div>
        
        <div 
             onClick={onHome}
             className="flex items-center gap-4 text-[#b3b3b3] font-bold cursor-pointer hover:text-white transition-colors"
        >
            <Search size={24} />
            <span>Search</span>
        </div>
      </div>

      <div className="bg-[#121212] rounded-lg p-4 flex-1 flex flex-col gap-4 mt-2">
        <div 
            onClick={() => setView('liked')}
            className={`flex items-center gap-4 font-bold cursor-pointer transition-colors ${view === 'liked' ? 'text-white' : 'text-[#b3b3b3] hover:text-white'}`}
        >
            <div className="bg-gradient-to-br from-indigo-700 to-blue-300 p-1 rounded-sm flex items-center justify-center opacity-70"><Heart size={16} className="text-white fill-white" /></div>
            <span>Liked Songs</span>
        </div>
        
        <div onClick={onRetrain} className="flex items-center gap-4 text-[#b3b3b3] font-bold cursor-pointer hover:text-white transition-colors mt-2">
            <span className="text-xl">✨</span>
            <span className="text-sm">Improve Recommendations</span>
        </div>
        
        <div className="border-t border-[#282828] mt-2 pt-4">
             <div className="p-2 text-xs text-[#b3b3b3]">
                <p>Model: <span className="text-white font-mono">{modelVersion}</span></p>
                <p>Status: Active</p>
             </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
