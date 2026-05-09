import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom'
import HomePage from './pages/HomePage'
import MovieDetailPage from './pages/MovieDetailPage'
import ExplorePage from './pages/ExplorePage'
import SearchBar from './components/SearchBar'

function Navbar() {
  const { pathname } = useLocation()
  const links = [
    { to: '/', label: 'Trang chu' },
    { to: '/explore', label: 'Kham pha' },
  ]
  return (
    <nav className="bg-[#0f3460] border-b border-gray-700 sticky top-0 z-40">
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center gap-6">
        <Link to="/" className="text-[#e94560] font-bold text-lg shrink-0">
          🎬 MovieAI
        </Link>

        {/* Nav links */}
        <div className="hidden sm:flex gap-4">
          {links.map(l => (
            <Link
              key={l.to}
              to={l.to}
              className={`text-sm font-medium transition-colors
                ${pathname === l.to ? 'text-white' : 'text-gray-400 hover:text-white'}`}
            >
              {l.label}
            </Link>
          ))}
        </div>

        {/* Search - hidden on detail page */}
        {!pathname.startsWith('/movie/') && (
          <div className="flex-1 flex justify-end">
            <SearchBar />
          </div>
        )}
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main>
        <Routes>
          <Route path="/"           element={<HomePage />} />
          <Route path="/movie/:id"  element={<MovieDetailPage />} />
          <Route path="/explore"    element={<ExplorePage />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
