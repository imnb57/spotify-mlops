import { useState } from 'react'
import axios from 'axios'
import { motion } from 'framer-motion'
import { Search, Music, Disc, PlayCircle } from 'lucide-react'

const API_URL = 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [selectedSong, setSelectedSong] = useState(null)
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(false)

  const searchSongs = async (e) => {
    e.preventDefault()
    if (!query) return;
    setLoading(true)
    try {
      const res = await axios.get(`${API_URL}/search?q=${query}&limit=5`)
      setSearchResults(res.data)
      setRecommendations([])
      setSelectedSong(null)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const getRecommendations = async (song) => {
    setSelectedSong(song)
    setLoading(true)
    try {
      const res = await axios.get(`${API_URL}/recommend/${song.track_id}`)
      setRecommendations(res.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-spotify-black text-white p-8 font-sans">
      <div className="max-w-4xl mx-auto">
        <header className="mb-12 text-center">
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-5xl font-bold mb-4 text-spotify-green tracking-tight"
          >
            Spotify ML Recommender
          </motion.h1>
          <p className="text-spotify-grey text-lg">Discover your next favorite track with AI</p>
        </header>

        {/* Search Section */}
        <div className="mb-12">
          <form onSubmit={searchSongs} className="relative max-w-2xl mx-auto">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for a song..."
              className="w-full bg-spotify-lightd rounded-full py-4 px-6 pl-12 text-lg focus:outline-none focus:ring-2 focus:ring-spotify-green transition-all shadow-lg placeholder-spotify-grey"
            />
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-spotify-grey" size={24} />
            <button 
              type="submit"
              className="absolute right-2 top-2 bottom-2 bg-spotify-green text-black font-bold px-6 rounded-full hover:bg-green-400 transition-transform hover:scale-105"
            >
              Search
            </button>
          </form>
        </div>

        {/* Results & Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          
          {/* Search Results */}
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-4"
          > 
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Music className="text-spotify-green" /> Results
            </h2>
            {searchResults.length === 0 && !loading && (
              <p className="text-spotify-grey italic ml-2">No active search...</p>
            )}
            {searchResults.map((song) => (
              <motion.div
                key={song.track_id}
                whileHover={{ scale: 1.02 }}
                onClick={() => getRecommendations(song)}
                className={`p-4 rounded-xl cursor-pointer transition-colors flex items-center gap-4 ${
                  selectedSong?.track_id === song.track_id 
                    ? 'bg-spotify-green text-black' 
                    : 'bg-spotify-lightd hover:bg-[#3E3E3E]'
                }`}
              >
                <div className={`p-3 rounded-full ${selectedSong?.track_id === song.track_id ? 'bg-black/20' : 'bg-spotify-dark'}`}>
                   <Music size={20} />
                </div>
                <div>
                  <h3 className="font-bold text-lg">{song.track_name}</h3>
                  <p className={`text-sm ${selectedSong?.track_id === song.track_id ? 'text-black/70' : 'text-spotify-grey'}`}>
                    {song.artists}
                  </p>
                </div>
              </motion.div>
            ))}
          </motion.div>

          {/* Recommendations */}
          <motion.div 
             initial={{ opacity: 0 }}
             animate={{ opacity: 1 }}
             className="space-y-4"
          >
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Disc className="text-spotify-green" /> Recommendations
            </h2>
            
            {!selectedSong && (
               <p className="text-spotify-grey italic ml-2">Select a song to get recommendations</p>
            )}

            {recommendations.map((song, idx) => (
              <motion.div
                key={song.track_id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="p-4 rounded-xl bg-gradient-to-r from-spotify-lightd to-[#2a2a2a] hover:from-[#3E3E3E] hover:to-[#3E3E3E] flex items-center gap-4 transform transition-all group"
              >
                <div className="text-spotify-green font-bold text-xl w-6">
                  {idx + 1}
                </div>
                <div className="flex-1">
                  <h3 className="font-bold text-lg group-hover:text-spotify-green transition-colors">{song.track_name}</h3>
                  <p className="text-spotify-grey text-sm">{song.artists}</p>
                </div>
                <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                   <PlayCircle className="text-spotify-green" size={32} />
                </div>
              </motion.div>
            ))}
          </motion.div>

        </div>
      </div>
    </div>
  )
}

export default App
