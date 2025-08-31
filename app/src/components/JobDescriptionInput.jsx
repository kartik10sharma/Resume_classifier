import { useState } from 'react';

const JobDescriptionInput = ({ value, onChange }) => {
  const [isFocused, setIsFocused] = useState(false);

  const styles = {
    container: {
      marginBottom: '24px'
    },
    label: {
      display: 'block',
      fontSize: '14px',
      fontWeight: '600',
      color: '#374151',
      marginBottom: '8px'
    },
    textarea: {
      width: '100%',
      padding: '16px',
      border: '1px solid #d1d5db',
      borderRadius: '12px',
      fontSize: '14px',
      lineHeight: '1.5',
      resize: 'vertical',
      minHeight: '140px',
      backgroundColor: '#ffffff',
      transition: 'all 0.3s ease',
      outline: 'none',
      boxSizing: 'border-box'
    },
    textareaFocus: {
      borderColor: '#3b82f6',
      boxShadow: '0 0 0 3px rgba(59, 130, 246, 0.1)'
    }
  };

  return (
    <div style={styles.container}>
      <label style={styles.label}>Job Description</label>
      <textarea
        value={value}
        onChange={onChange}
        placeholder="Paste the job description here..."
        style={{
          ...styles.textarea,
          ...(isFocused ? styles.textareaFocus : {})
        }}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
      />
    </div>
  );
};

export default JobDescriptionInput;