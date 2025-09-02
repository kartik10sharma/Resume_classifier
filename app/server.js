import express from "express";
import multer from "multer";
import pkg from "pg";
import dotenv from "dotenv";
import cors from "cors";

dotenv.config();
const { Pool } = pkg;

const app = express();
const upload = multer(); // in-memory storage

// ✅ Enable CORS so frontend (5173) can talk to backend (4000)
app.use(cors());

// ✅ Parse JSON (if needed for other routes)
app.use(express.json());

const pool = new Pool({
  connectionString: process.env.DATABASE_URL, // from NeonDB
  ssl: { rejectUnauthorized: false }
});

// Upload resume endpoint
app.post("/upload", upload.single("resume"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: "No file uploaded" });
    }

    const { originalname, mimetype, buffer } = req.file;

    await pool.query(
      "INSERT INTO resumes (filename, mimetype, file_data) VALUES ($1, $2, $3)",
      [originalname, mimetype, buffer]
    );

    res.json({ success: true, message: "Resume stored in NeonDB!" });
  } catch (err) {
    console.error("Upload error:", err.message);
    res.status(500).json({ success: false, error: "Upload failed" });
  }
});

// Run server on port 4000
app.listen(4000, () => {
  console.log("✅ Server running on http://localhost:4000");
});
