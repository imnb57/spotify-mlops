import { useState,useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Sliders } from 'lucide-react';
import Layout from './components/Layout';
import Header from './components/Header';
import SongCard from './components/SongCard';
import Toast from './components/Toast';
import Tuner from './components/Tuner';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedSong, setSelectedSong] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [likedSongs, setLikedSongs] = useState(() => {
    const saved = localStorage.getItem('likedSongsList');
    if (saved) {
        try {
            const parsed = JSON.parse(saved);
            return new Set(parsed.map(s => s.track_id));
        } catch(e) { console.error(e); }
    }
    return new Set();
  }); 
  const [likedSongsList, setLikedSongsList] = useState(() => {
    const saved = localStorage.getItem('likedSongsList');
    if (saved) {
        try {
            return JSON.parse(saved);
        } catch(e) { console.error(e); }
    }
    return [];
  });
  
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState('home'); 
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');

  const [homeTitle, setHomeTitle] = useState('Made For You');
  
  // V5 State
  const [showTuner, setShowTuner] = useState(false);
  const [tunerValues, setTunerValues] = useState({ energy: 0, danceability: 0, valence: 0, acousticness: 0 });

  useEffect(() => {
    fetchHomeFeed();
  }, [tunerValues]); // Re-fetch when tuner values change

  // Save likes to localStorage whenever they change
  useEffect(() => {
      localStorage.setItem('likedSongsList', JSON.stringify(likedSongsList));
  }, [likedSongsList]);

  const fetchHomeFeed = async () => {
      try {
          // Construct Query Params from Tuner
          const params = new URLSearchParams({ limit: 12 });
          if(tunerValues.energy !== 0) params.append('target_energy', tunerValues.energy);
          if(tunerValues.danceability !== 0) params.append('target_danceability', tunerValues.danceability);
          if(tunerValues.valence !== 0) params.append('target_valence', tunerValues.valence);
          if(tunerValues.acousticness !== 0) params.append('target_acousticness', tunerValues.acousticness);

          const res = await axios.get(`${API_URL}/recommend/home?${params.toString()}`);
          setRecommendations(res.data);
          
          // Check if we have reasons to determine title
          if (res.data.length > 0 && res.data[0].reason === 'Trending Now') {
              setHomeTitle("Trending Now");
          } else {
              setHomeTitle("Made For You");
          }
      } catch (err) {
          console.error("Home Feed Error:", err);
      }
  };

  const startRadio = async (song) => {
      setLoading(true);
      setHomeTitle(`Radio: ${song.track_name}`);
      setView('home');
      setRecommendations([]); // clear old
      setSearchResults([]); // clear search results if any
      setQuery(''); // clear query
      
      try {
          const res = await axios.get(`${API_URL}/recommend/radio/${song.track_id}?limit=12`);
          setRecommendations(res.data);
          setToastMessage(`Started radio for "${song.track_name}"`);
          setShowToast(true);
      } catch(e) {
          console.error("Radio Error", e);
      } finally {
          setLoading(false);
      }
  };

  const handleSearch = async (e) => {
    e?.preventDefault();
    if (!query) {
       setView('home'); 
       fetchHomeFeed();
       return;
    }
    setLoading(true);
    setView('home'); 
    try {
      const res = await axios.get(`${API_URL}/search?q=${query}&limit=6`);
      setSearchResults(res.data);
      setRecommendations([]);
    } catch (err) {
      console.error("Search Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const playSong = async (song) => {
    setSelectedSong(song);
  };

  const handleLike = async (song) => {
    const isLiked = likedSongs.has(song.track_id);
    const newLiked = new Set(likedSongs);
    
    if (isLiked) {
        newLiked.delete(song.track_id);
        const newList = likedSongsList.filter(s => s.track_id !== song.track_id);
        setLikedSongsList(newList);
    } else {
        newLiked.add(song.track_id);
        setLikedSongsList([...likedSongsList, song]);
    }
    setLikedSongs(newLiked);

    try {
        await axios.post(`${API_URL}/feedback`, {
            track_id: song.track_id,
            liked: !isLiked
        });
    } catch (err) {
        console.error("Feedback error:", err);
    }
  };
  
  const handleRetrain = async () => {
    try {
        setToastMessage("Training model based on your likes...");
        setShowToast(true);
        const res = await axios.post(`${API_URL}/control/retrain`);
        if (res.data.status === 'success') {
             setTimeout(() => {
                 setToastMessage("Model Updated! Recommendations improved.");
                 setShowToast(true);
             }, 5000); 
        }
    } catch (err) {
        console.error("Retrain Error:", err);
        setToastMessage("Training failed.");
        setShowToast(true);
    }
  };

  const currentViewContent = () => {
      if (view === 'liked') {
          return (
            <section>
              <h3 className="text-2xl font-bold text-white mb-4">Liked Songs</h3>
              {likedSongsList.length === 0 ? (
                  <p className="text-[#a7a7a7]">No liked songs yet.</p>
              ) : (
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-6">
                    {likedSongsList.map((song) => (
                      <SongCard 
                        key={song.track_id} 
                        song={song} 
                        onClick={() => playSong(song)} 
                        onLike={handleLike}
                        onRadio={() => startRadio(song)}
                        isActive={selectedSong?.track_id === song.track_id}
                        isLiked={true}
                      />
                    ))}
                  </div>
              )}
            </section>
          );
      }

      // Default Home View
      return (
        <>
            {/* Search Results */}
            {searchResults.length > 0 && (
                <section className="mb-8">
                  <h3 className="text-2xl font-bold text-white mb-4">Top Results</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-6">
                    {searchResults.map((song) => (
                      <SongCard 
                        key={song.track_id} 
                        song={song} 
                        onClick={() => playSong(song)}
                        onLike={handleLike}
                        onRadio={() => startRadio(song)}
                        isActive={selectedSong?.track_id === song.track_id}
                        isLiked={likedSongs.has(song.track_id)}
                      />
                    ))}
                  </div>
                  {searchResults.length === 6 && (
                    <div className="mt-8 flex justify-center pb-4">
                        <button 
                            onClick={async () => {
                                setLoading(true);
                                try {
                                    const res = await axios.get(`${API_URL}/search?q=${query}&limit=50`);
                                    setSearchResults(res.data);
                                } catch(e) { console.error(e); }
                                finally { setLoading(false); }
                            }}
                            className="bg-transparent border border-[#727272] hover:border-white text-white text-xs font-bold uppercase tracking-widest px-8 py-3 rounded-full transition-all hover:scale-105"
                        >
                            See All Results
                        </button>
                    </div>
                  )}
                </section>
            )}

            {/* Recommendations */}
            {recommendations.length > 0 && (
                <section>
                  <div className="flex items-center gap-4 mb-4">
                      <h3 className="text-2xl font-bold text-white">{homeTitle}</h3>
                      {view === 'home' && (
                          <button 
                            onClick={() => setShowTuner(!showTuner)}
                            className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider transition-colors border ${showTuner ? 'bg-spotify-green text-black border-spotify-green' : 'bg-transparent text-[#b3b3b3] border-[#727272] hover:text-white'}`}
                          >
                             <Sliders size={14} /> Tune
                          </button>
                      )}
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-6">
                    {recommendations.map((song) => (
                      <SongCard 
                        key={song.track_id} 
                        song={song} 
                        onClick={() => playSong(song)} 
                        onLike={handleLike}
                        onRadio={() => startRadio(song)}
                        isActive={selectedSong?.track_id === song.track_id}
                        isLiked={likedSongs.has(song.track_id)}
                      />
                    ))}
                  </div>
                </section>
            )}

            {!loading && searchResults.length === 0 && recommendations.length === 0 && (
                 <div className="flex flex-col items-center justify-center mt-20 text-[#a7a7a7]">
                    <span className="text-6xl mb-4">🎹</span>
                    <p className="text-xl">Search for a song or like some music to get started.</p>
                 </div>
            )}
        </>
      );
  };

  const resetToHome = () => {
      setQuery('');
      setSearchResults([]);
      setView('home');
      fetchHomeFeed(); 
  };

  return (
    <Layout currentSong={selectedSong} view={view} setView={setView} onRetrain={handleRetrain} onHome={resetToHome}>
      <div className="bg-gradient-to-b from-[#1e1e1e] to-[#121212] min-h-full relative">
        <Header search={query} setSearch={setQuery} onSearch={handleSearch} />
        
        <Tuner 
            isOpen={showTuner} 
            onClose={() => setShowTuner(false)}
            onUpdate={setTunerValues}
            currentValues={tunerValues}
        />
        
        {/* Notifications */}
        <AnimatePresence>
            {showToast && (
                <Toast 
                    message={toastMessage} 
                    type="loading" 
                    onClose={() => setShowToast(false)} 
                />
            )}
        </AnimatePresence>
        
        <div className="p-8 pb-32">
          
          {/* Welcome / Status */}
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
             <h2 className="text-3xl font-bold text-white mb-2">
                 {view === 'home' ? "Good afternoon" : "Your Library"}
             </h2>
             <p className="text-[#a7a7a7] text-sm md:text-base">
                 {view === 'home' ? "AI-Powered recommendations ready." : "Songs you liked."}
             </p>
          </motion.div>
            
          {currentViewContent()}
          
          {loading && (
             <div className="flex items-center justify-center mt-10">
                 <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-spotify-green"></div>
             </div>
          )}

        </div>
      </div>
    </Layout>
  );
}

export default App;
