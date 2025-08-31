import { useState } from 'react';

const SubmitButton = ({ onClick, disabled }) => {
  const [isHovered, setIsHovered] = useState(false);

  const styles = {
    button: {
      width: '100%',
      padding: '16px',
      backgroundColor: disabled ? '#9ca3af' : (isHovered ? '#2563eb' : '#3b82f6'),
      color: 'white',
      border: 'none',
      borderRadius: '12px',
      fontSize: '16px',
      fontWeight: '600',
      cursor: disabled ? 'not-allowed' : 'pointer',
      transition: 'all 0.3s ease',
      transform: isHovered && !disabled ? 'translateY(-2px)' : 'translateY(0)',
      boxShadow: isHovered && !disabled ? '0 8px 25px rgba(59, 130, 246, 0.3)' : '0 4px 12px rgba(0, 0, 0, 0.1)'
    }
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={styles.button}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {disabled ? 'Please fill all fields' : '🚀 Analyze Application'}
    </button>
  );
};

export default SubmitButton;