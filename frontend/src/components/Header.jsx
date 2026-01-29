import { Search, ChevronLeft, ChevronRight, User } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

const Header = ({ search, setSearch, onSearch }) => {
  const [suggestions, setSuggestions] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const wrapperRef = useRef(null);

  // Debounce search for dropdown
  useEffect(() => {
    const timer = setTimeout(async () => {
        if (search.length > 2) {
            try {
                const res = await axios.get(`${API_URL}/search?q=${search}&limit=5`);
                setSuggestions(res.data);
                setShowDropdown(true);
            } catch (err) {
                console.error(err);
            }
        } else {
            setSuggestions([]);
            setShowDropdown(false);
        }
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  // Click outside to close
  useEffect(() => {
      function handleClickOutside(event) {
          if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
            setShowDropdown(false);
          }
      }
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [wrapperRef]);
  
  const handleSelect = (song) => {
      setSearch(song.track_name);
      setShowDropdown(false);
      // Trigger search with this new value
      // We need to call onSearch but it expects an event usually?
      // App.jsx handleSearch checks `query` state or event?
      // App.jsx handleSearch checks `query` state.
      // We just updated search (which updates query in App).
      // We can mock an event for onSearch.
      onSearch({ preventDefault: () => {} }); 
  };

  return (
    <div className="h-16 bg-[#090909] flex items-center justify-between px-8 sticky top-0 z-20 opacity-95">
      
      {/* Navigation History Mock */}
      <div className="flex items-center gap-4 hidden md:flex">
        <div className="bg-[#121212] p-1 rounded-full cursor-not-allowed opacity-50"><ChevronLeft size={24} color="white" /></div>
        <div className="bg-[#121212] p-1 rounded-full cursor-not-allowed opacity-50"><ChevronRight size={24} color="white" /></div>
      </div>
      
      {/* Search Bar */}
      <div className="relative flex-1 max-w-[400px] ml-4" ref={wrapperRef}>
         <div className="bg-[#242424] rounded-full flex items-center px-4 py-3 hover:bg-[#2a2a2a] group focus-within:ring-2 focus-within:ring-white/20 transition-all">
            <Search size={22} className="text-[#b3b3b3] group-focus-within:text-white transition-colors" />
            <form onSubmit={(e) => { setShowDropdown(false); onSearch(e); }} className="w-full">
                <input 
                    type="text" 
                    placeholder="What do you want to play?" 
                    className="bg-transparent border-none outline-none text-white ml-3 w-full placeholder-[#757575] font-medium text-sm"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    onFocus={() => { if(suggestions.length > 0) setShowDropdown(true); }}
                />
            </form>
         </div>

         {/* Autocomplete Dropdown */}
         {showDropdown && suggestions.length > 0 && (
            <div className="absolute top-14 left-0 w-full bg-[#242424] rounded-lg shadow-2xl overflow-hidden py-2 z-50 animate-in fade-in zoom-in-95 duration-200">
                <h4 className="px-4 py-2 text-xs font-bold text-[#b3b3b3] uppercase tracking-wider">Top Suggestions</h4>
                {suggestions.map((song) => (
                    <div 
                        key={song.track_id}
                        onClick={() => handleSelect(song)}
                        className="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer flex items-center gap-3 transition-colors"
                    >
                         {song.album_cover ? (
                             <img src={song.album_cover} className="w-10 h-10 rounded-sm object-cover shadow-sm" />
                         ) : (
                             <div className="w-10 h-10 bg-gray-700 flex items-center justify-center rounded-sm">🎵</div>
                         )}
                         <div className="flex flex-col overflow-hidden">
                             <span className="text-sm text-white truncate font-medium">{song.track_name}</span>
                             <span className="text-xs text-[#b3b3b3] truncate">{song.artists}</span>
                         </div>
                    </div>
                ))}
            </div>
         )}
      </div>

      {/* User / Actions */}
      <div className="flex items-center gap-4 ml-4">
        <button className="text-[#a7a7a7] text-sm font-bold hover:text-white hover:scale-105 transition-all hidden sm:block">Sign up</button>
        <button className="bg-white text-black font-bold text-sm px-6 py-3 rounded-full hover:scale-105 transition-all hidden sm:block">Log in</button>
        <div className="bg-[#121212] p-2 rounded-full cursor-pointer hover:bg-[#2a2a2a]" title="User Profile">
            <User className="text-white" size={20} />
        </div>
      </div>

    </div>
  );
};

export default Header;
