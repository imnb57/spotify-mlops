import { useState } from 'react';
import { Sliders, X, RotateCcw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Tuner = ({ isOpen, onClose, onUpdate, currentValues }) => {
  const [values, setValues] = useState(currentValues);

  const handleChange = (feature, val) => {
      const newValues = { ...values, [feature]: parseFloat(val) };
      setValues(newValues);
  };

  const handleApply = () => {
      onUpdate(values);
  };
  
  const handleReset = () => {
      const defaults = { energy: 0, danceability: 0, valence: 0, acousticness: 0 };
      setValues(defaults);
      onUpdate(defaults);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div 
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-[#181818] border-b border-[#282828] overflow-hidden"
        >
          <div className="p-8 max-w-4xl mx-auto">
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-white text-lg font-bold flex items-center gap-2">
                    <Sliders size={20} className="text-spotify-green"/>
                    Sonic Tuner
                </h3>
                <div className="flex gap-4">
                     <button onClick={handleReset} className="text-[#b3b3b3] hover:text-white text-xs uppercase tracking-widest font-bold flex items-center gap-1">
                        <RotateCcw size={14}/> Reset
                     </button>
                     <button onClick={onClose} className="text-[#b3b3b3] hover:text-white">
                        <X size={24}/>
                     </button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6">
                {/* Energy */}
                <div className="space-y-2">
                    <div className="flex justify-between text-xs font-bold uppercase tracking-wider text-[#b3b3b3]">
                        <span>Chill</span>
                        <span className="text-white">Energy</span>
                        <span>Intense</span>
                    </div>
                    <input 
                        type="range" min="-1.5" max="1.5" step="0.1"
                        value={values.energy || 0}
                        onChange={(e) => handleChange('energy', e.target.value)}
                        className="w-full h-1 bg-[#404040] rounded-lg appearance-none cursor-pointer accent-spotify-green"
                    />
                </div>

                {/* Danceability */}
                <div className="space-y-2">
                    <div className="flex justify-between text-xs font-bold uppercase tracking-wider text-[#b3b3b3]">
                        <span>Static</span>
                        <span className="text-white">Dance</span>
                        <span>Groovy</span>
                    </div>
                    <input 
                        type="range" min="-1.5" max="1.5" step="0.1"
                        value={values.danceability || 0}
                        onChange={(e) => handleChange('danceability', e.target.value)}
                        className="w-full h-1 bg-[#404040] rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                </div>
                
                 {/* Valence (Mood) */}
                <div className="space-y-2">
                    <div className="flex justify-between text-xs font-bold uppercase tracking-wider text-[#b3b3b3]">
                        <span>Sad</span>
                        <span className="text-white">Mood</span>
                        <span>Happy</span>
                    </div>
                    <input 
                        type="range" min="-1.5" max="1.5" step="0.1"
                        value={values.valence || 0}
                        onChange={(e) => handleChange('valence', e.target.value)}
                        className="w-full h-1 bg-[#404040] rounded-lg appearance-none cursor-pointer accent-yellow-500"
                    />
                </div>
                
                 {/* Acousticness */}
                <div className="space-y-2">
                    <div className="flex justify-between text-xs font-bold uppercase tracking-wider text-[#b3b3b3]">
                        <span>Electric</span>
                        <span className="text-white">Texture</span>
                        <span>Acoustic</span>
                    </div>
                    <input 
                        type="range" min="-1.5" max="1.5" step="0.1"
                        value={values.acousticness || 0}
                        onChange={(e) => handleChange('acousticness', e.target.value)}
                        className="w-full h-1 bg-[#404040] rounded-lg appearance-none cursor-pointer accent-orange-500"
                    />
                </div>
            </div>

            <div className="flex justify-center">
                <button 
                    onClick={handleApply}
                    className="bg-white text-black font-bold uppercase tracking-widest text-xs px-8 py-3 rounded-full hover:scale-105 transition-transform"
                >
                    Apply Settings
                </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default Tuner;
