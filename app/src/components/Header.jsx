const Header = () => {
  const styles = {
    header: {
      textAlign: 'center',
      marginBottom: '32px'
    },
    title: {
      fontSize: '28px',
      fontWeight: '700',
      color: '#111827',
      marginBottom: '8px'
    },
    subtitle: {
      fontSize: '16px',
      color: '#6b7280'
    }
  };

  return (
    <div style={styles.header}>
      <h1 style={styles.title}>Resume Analyzer</h1>
      <p style={styles.subtitle}>Upload your resume and job description for analysis</p>
    </div>
  );
};

export default Header;