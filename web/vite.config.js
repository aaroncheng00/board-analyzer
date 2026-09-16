import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The /api proxy keeps the browser talking to a single origin, so the API needs
// no CORS middleware.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
})
