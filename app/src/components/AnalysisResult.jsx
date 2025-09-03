import React from "react";

const AnalysisResult = ({ analysisResult }) => {
  if (!analysisResult) return null;

  const { analysis, file_processed, recommendation, level, probability_percent } =
    analysisResult;

  //  Styles 
  const styles = {
    card: {
      marginTop: "20px",
      padding: "24px",
      background: "rgba(255, 255, 255, 0.9)",
      borderRadius: "16px",
      boxShadow: "0 6px 18px rgba(0, 0, 0, 0.1)",
      border: "1px solid #e0e0e0",
      fontFamily: "Segoe UI, Tahoma, sans-serif",
      animation: "fadeIn 0.6s ease-in-out",
    },
    title: {
      fontSize: "20px",
      fontWeight: 600,
      color: "#333",
      borderBottom: "2px solid #d0d7ff",
      paddingBottom: "6px",
      marginBottom: "14px",
      display: "flex",
      alignItems: "center",
      gap: "6px",
    },
    section: {
      background: "#f9f9fc",
      border: "1px solid #eee",
      borderRadius: "12px",
      padding: "12px 16px",
      marginBottom: "16px",
    },
    label: {
      fontWeight: 600,
      color: "#555",
      marginBottom: "4px",
    },
    text: {
      color: "#444",
      margin: "2px 0",
    },
    highlight: {
      fontWeight: "bold",
      color: "#4a61ff",
    },
    skillBar: {
      margin: "10px 0",
    },
    skillInfo: {
      display: "flex",
      justifyContent: "space-between",
      fontSize: "14px",
      marginBottom: "4px",
      color: "#444",
    },
    progress: {
      background: "#e6e6e6",
      height: "8px",
      borderRadius: "6px",
      overflow: "hidden",
    },
    progressFill: (value) => ({
      height: "8px",
      width: `${value}%`,
      background: "linear-gradient(90deg, #4a61ff, #00c6ff)",
      borderRadius: "6px",
      transition: "width 0.4s ease",
    }),
    list: {
      listStyle: "disc",
      paddingLeft: "20px",
      color: "#444",
    },
  };

  return (
    <div style={styles.card}>
      <h3 style={styles.title}>📊 Analysis Result</h3>

      {/* File Info */}
      <div style={styles.section}>
        <p style={styles.label}>📄 File Processed:</p>
        <p style={styles.text}>{file_processed}</p>
      </div>

      {/* Score Section */}
      <div style={styles.section}>
        <p style={styles.label}>🎯 Predicted Score:</p>
        <p style={styles.text}>
          <span style={styles.highlight}>{analysis?.predicted_score}</span> (
          {(analysis?.confidence * 100).toFixed(1)}% confidence)
        </p>
        <p style={styles.text}>
          Match Probability:{" "}
          <span style={styles.highlight}>
            {(analysis?.match_probability * 100).toFixed(1)}%
          </span>
        </p>
        <p style={styles.text}>
          Level: <span style={styles.highlight}>{level}</span> | Probability:{" "}
          <span style={styles.highlight}>{probability_percent}%</span>
        </p>
      </div>

      {/* Skill Matches */}
      <div style={styles.section}>
        <p style={styles.label}>🛠 Skill Analysis</p>
        {Object.entries(analysis?.skill_analysis || {}).map(([skill, value]) => (
          <div key={skill} style={styles.skillBar}>
            <div style={styles.skillInfo}>
              <span>{skill.replace(/_/g, " ")}</span>
              <span>{(value * 100).toFixed(0)}%</span>
            </div>
            <div style={styles.progress}>
              <div style={styles.progressFill(value * 100)} />
            </div>
          </div>
        ))}
      </div>

      {/* Recommendation */}
      <div style={styles.section}>
        <p style={styles.label}>✅ Recommendation</p>
        <p style={styles.text}>{recommendation?.action}</p>
        {recommendation?.improvement_areas?.length > 0 && (
          <>
            <p style={styles.label}>📌 Improvement Areas:</p>
            <ul style={styles.list}>
              {recommendation.improvement_areas.map((area, idx) => (
                <li key={idx}>{area}</li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
};

export default AnalysisResult;
