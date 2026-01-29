import Sidebar from './Sidebar';
import Player from './Player';

const Layout = ({ children, currentSong, view, setView, onRetrain, onHome }) => {
  return (
    <div className="flex flex-col h-screen bg-black overflow-hidden font-sans">
      <div className="flex flex-1 overflow-hidden">
        <Sidebar view={view} setView={setView} onRetrain={onRetrain} onHome={onHome} />
        <main className="flex-1 rounded-lg bg-[#121212] ml-2 mb-2 mr-2 overflow-y-auto relative no-scrollbar">
          {children}
        </main>
      </div>
      <Player currentSong={currentSong} />
    </div>
  );
};

export default Layout;
