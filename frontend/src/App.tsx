import { Routes, Route } from 'react-router-dom'
import { Analytics } from '@vercel/analytics/react'
import GraphView from '@/views/GraphView'
import './App.css'

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<GraphView />} />
      </Routes>
      <Analytics />
    </div>
  )
}

export default App
