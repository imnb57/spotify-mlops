import { motion } from 'framer-motion';
import { Play, Heart, Radio } from 'lucide-react';

const SongCard = ({ song, onClick, onLike, onRadio, isActive, isLiked }) => {
  return (
    <motion.div
      whileHover={{ y: -5 }}
      className={`p-4 rounded-md cursor-pointer transition-all group relative bg-[#181818] hover:bg-[#282828] ${
        isActive ? 'bg-[#282828]' : ''
      }`}
    >
       {/* Card Click Area */}
      <div onClick={onClick}>
          <div className="relative mb-4 shadow-lg rounded-md overflow-hidden aspect-square bg-[#333]">
            {song.album_cover ? (
              <img src={song.album_cover} alt={song.album_name} className="w-full h-full object-cover" />
            ) : (
                <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-800 to-black text-gray-500">
                    <span className="text-4xl">🎵</span>
                </div>
            )}
            
            {/* Play Button */}
            <div className={`absolute bottom-2 right-2 rounded-full bg-spotify-green p-3 shadow-xl transform translate-y-1/4 opacity-0 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-300 flex items-center justify-center`}>
              <Play fill="black" className="text-black ml-1" size={20} />
            </div>
            
             {/* Radio Button (Top Left) */}
            <div className="absolute top-2 left-2 bg-black/60 rounded-full p-2 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/90 z-20" onClick={(e) => { e.stopPropagation(); onRadio(song); }} title="Start Song Radio">
                 <Radio size={16} className="text-white" />
            </div>

          </div>
          
          <h3 className="font-bold text-white text-base truncate mb-1">{song.track_name}</h3>
          <p className="text-[#a7a7a7] text-sm truncate line-clamp-2">{song.artists}</p>
          {song.reason && (
            <div className="mt-2 inline-block bg-[#282828] text-spotify-green text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">
                {song.reason}
            </div>
          )}
      </div>

      {/* Like Button (Separate Action) */}
      <div className="absolute top-4 right-4 bg-black/50 rounded-full p-2 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/80 z-10" onClick={(e) => { e.stopPropagation(); onLike(song); }}>
          <Heart size={18} className={`${isLiked ? 'text-spotify-green fill-spotify-green' : 'text-white'}`} />
      </div>

    </motion.div>
  );
};

export default SongCard;
