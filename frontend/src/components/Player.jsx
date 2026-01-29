import { useState, useEffect } from 'react';
import { PlayCircle, PauseCircle, SkipBack, SkipForward, Repeat, Shuffle, Volume2, VolumeX } from 'lucide-react';

const Player = ({ currentSong }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(70);
  const [progress, setProgress] = useState(0);

  // Reset state when song changes
  useEffect(() => {
    if (currentSong) {
        setIsPlaying(true);
        setProgress(0);
    } else {
        setIsPlaying(false);
    }
  }, [currentSong]);

  // Mock progress interval
  useEffect(() => {
    let interval;
    if (isPlaying && currentSong) {
        interval = setInterval(() => {
            setProgress((prev) => (prev >= 100 ? 0 : prev + 1));
        }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, currentSong]);

  if (!currentSong) return (
     <div className="h-24 bg-[#181818] border-t border-[#282828] text-white flex items-center justify-center z-50 fixed bottom-0 w-full">
        <p className="text-[#a7a7a7] text-sm font-medium">Select a song to start listening</p>
     </div>
  );

  return (
    <div className="h-24 bg-[#181818] border-t border-[#282828] px-4 flex items-center justify-between z-50 fixed bottom-0 w-full">
      
      {/* Song Info */}
      <div className="flex items-center gap-4 w-[30%]">
        <div className="w-14 h-14 bg-[#333] shadow-lg rounded-sm overflow-hidden flex-shrink-0">
             {currentSong.album_cover ? (
                <img src={currentSong.album_cover} alt="" className="w-full h-full object-cover" />
             ) : (
                <div className="w-full h-full bg-gray-800 flex items-center justify-center text-xl">🎵</div>
             )}
        </div>
        <div className="flex flex-col justify-center overflow-hidden">
            <h4 className="text-white text-sm font-semibold hover:underline cursor-pointer truncate">{currentSong.track_name}</h4>
            <p className="text-[#b3b3b3] text-xs hover:underline cursor-pointer truncate">{currentSong.artists}</p>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-col items-center w-[40%] gap-2">
        <div className="flex items-center gap-6">
            <Shuffle size={16} className="text-[#b3b3b3] hover:text-white cursor-pointer transition-colors" />
            <div className="active:scale-95 transition-transform"><SkipBack size={20} className="text-[#b3b3b3] hover:text-white cursor-pointer fill-current" /></div>
            
            <div onClick={() => setIsPlaying(!isPlaying)} className="bg-white rounded-full p-2 hover:scale-105 transition-transform cursor-pointer shadow-md">
                {isPlaying ? (
                    <PauseCircle size={24} className="text-black fill-black" />
                ) : (
                    <PlayCircle size={24} className="text-black fill-black" />
                )}
            </div>
            
            <div className="active:scale-95 transition-transform"><SkipForward size={20} className="text-[#b3b3b3] hover:text-white cursor-pointer fill-current" /></div>
            <Repeat size={16} className="text-[#b3b3b3] hover:text-white cursor-pointer transition-colors" />
        </div>
        
        {/* Progress Bar */}
        <div className="w-full flex items-center gap-2 text-xs text-[#a7a7a7]">
            <span>{Math.floor((progress/100) * 3)}:{(Math.floor((progress/100) * 60) % 60).toString().padStart(2, '0')}</span>
            <div className="h-1 bg-[#4d4d4d] rounded-full flex-1 group cursor-pointer relative">
                <div 
                    className="h-full bg-white rounded-full group-hover:bg-spotify-green relative transition-all duration-100 ease-linear"
                    style={{ width: `${progress}%` }}
                >
                     <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full opacity-0 group-hover:opacity-100 shadow-md transform scale-0 group-hover:scale-100 transition-all"></div>
                </div>
            </div>
            <span>3:45</span>
        </div>
      </div>

      {/* Volume */}
      <div className="flex items-center justify-end w-[30%] gap-2">
         {volume === 0 ? (
             <VolumeX size={18} className="text-[#b3b3b3] cursor-pointer" onClick={() => setVolume(70)} />
         ) : (
             <Volume2 size={18} className="text-[#b3b3b3] cursor-pointer" onClick={() => setVolume(0)} />
         )}
         
         <div className="w-24 h-1 bg-[#4d4d4d] rounded-full group cursor-pointer relative">
              <div 
                  className="h-full bg-[#b3b3b3] group-hover:bg-spotify-green rounded-full relative"
                  style={{ width: `${volume}%` }}
              >
                  <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full opacity-0 group-hover:opacity-100 shadow-md"></div>
              </div>
         </div>
      </div>

    </div>
  );
};

export default Player;
