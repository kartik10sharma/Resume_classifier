const FileUpload = ({ onFileChange, uploadedFile }) => {
  const styles = {
    container: {
      position: 'relative',
      marginBottom: '24px'
    },
    label: {
      display: 'block',
      fontSize: '14px',
      fontWeight: '600',
      color: '#374151',
      marginBottom: '8px'
    },
    fileInputWrapper: {
      position: 'relative',
      display: 'inline-block',
      width: '100%'
    },
    fileInput: {
      position: 'absolute',
      opacity: 0,
      width: '100%',
      height: '100%',
      cursor: 'pointer'
    },
    fileButton: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      width: '100%',
      padding: '20px',
      border: '2px dashed #d1d5db',
      borderRadius: '12px',
      backgroundColor: '#f9fafb',
      color: '#6b7280',
      fontSize: '14px',
      fontWeight: '500',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      boxSizing: 'border-box'
    },
    fileInfo: {
      marginTop: '12px',
      padding: '12px 16px',
      backgroundColor: '#ecfdf5',
      border: '1px solid #d1fae5',
      borderRadius: '8px',
      fontSize: '13px',
      color: '#065f46',
      display: 'flex',
      alignItems: 'center'
    },
    uploadIcon: {
      marginRight: '8px',
      fontSize: '18px'
    }
  };

  return (
    <div style={styles.container}>
      <label style={styles.label}>Upload Resume/Document</label>
      <div style={styles.fileInputWrapper}>
        <input
          type="file"
          onChange={onFileChange}
          style={styles.fileInput}
          accept=".pdf,.doc,.docx,.txt"
        />
        <div style={styles.fileButton}>
          <span style={styles.uploadIcon}>📄</span>
          {uploadedFile ? `Selected: ${uploadedFile.name}` : 'Click to upload or drag and drop'}
        </div>
      </div>
      {uploadedFile && (
        <div style={styles.fileInfo}>
          <span style={styles.uploadIcon}>✅</span>
          File uploaded: {uploadedFile.name} ({Math.round(uploadedFile.size / 1024)}KB)
        </div>
      )}
    </div>
  );
};

export default FileUpload;