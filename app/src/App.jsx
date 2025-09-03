import { useState } from 'react';
import Header from './components/Header';
import JobDescriptionInput from './components/JobDescriptionInput';
import FileUpload from './components/FileUpload';
import SubmitButton from './components/SubmitButton';
import AnalysisResult from './components/AnalysisResult'; // ✅ new import

const App = () => {
  const [jobDescription, setJobDescription] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setUploadedFile(file);
  };

  const handleJobDescriptionChange = (e) => {
    setJobDescription(e.target.value);
  };

  const handleSubmit = async () => {
    if (!jobDescription.trim() || !uploadedFile) return;

    try {
      setIsLoading(true);

      // Upload file
      const formDataUpload = new FormData();
      formDataUpload.append("resume", uploadedFile);

      await fetch("http://localhost:4000/upload", {
        method: "POST",
        body: formDataUpload,
      });

      // Analyze JD + Resume
      const formDataAnalyze = new FormData();
      formDataAnalyze.append("job_description", jobDescription);
      formDataAnalyze.append("resume", uploadedFile, uploadedFile.name);

      const response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        body: formDataAnalyze,
      });

      const result = await response.json();

      if (result.success) {
        setAnalysisResult(result);
        console.log("Analysis Result:", result);
      } else {
        alert("Error: " + result.error);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to analyze. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const isFormValid = jobDescription.trim() && uploadedFile;

  const styles = {
    app: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px'
    },
    container: {
      maxWidth: '700px',
      margin: '0 auto',
      backgroundColor: 'white',
      padding: '40px',
      borderRadius: '20px',
      boxShadow: '0 20px 40px rgba(0, 0, 0, 0.15)',
      border: '1px solid rgba(255, 255, 255, 0.2)',
      backdropFilter: 'blur(10px)'
    }
  };

  return (
    <div style={styles.app}>
      <div style={styles.container}>
        <Header />
        <JobDescriptionInput
          value={jobDescription}
          onChange={handleJobDescriptionChange}
        />
        <FileUpload
          onFileChange={handleFileChange}
          uploadedFile={uploadedFile}
        />
        <SubmitButton
          onClick={handleSubmit}
          disabled={!isFormValid || isLoading}
          label={isLoading ? 'Analyzing...' : 'Analyze Resume'}
        />
        <AnalysisResult analysisResult={analysisResult} />
      </div>
    </div>
  );
};

export default App;
