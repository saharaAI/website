import React, { useState } from 'react';

const Icon = ({ children, isActive }) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width="48" 
    height="48" 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke={isActive ? "#4F46E5" : "currentColor"} 
    strokeWidth="2" 
    strokeLinecap="round" 
    strokeLinejoin="round"
  >
    {children}
  </svg>
);

const BellIcon = () => (
  <Icon>
    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
    <path d="M13.73 21a2 2 0 0 1-3.46 0" />
  </Icon>
);

const GridIcon = () => (
  <Icon>
    <rect x="3" y="3" width="7" height="7" />
    <rect x="14" y="3" width="7" height="7" />
    <rect x="14" y="14" width="7" height="7" />
    <rect x="3" y="14" width="7" height="7" />
  </Icon>
);

const DashboardIcon = ({ icon: IconComponent, label, isActive, onClick }) => (
  <div 
    style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      padding: '1rem',
      cursor: 'pointer',
      color: isActive ? '#4F46E5' : 'inherit'
    }}
    onClick={onClick}
  >
    <IconComponent />
    <span style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>{label}</span>
  </div>
);

const Dashboard = () => {
  const [activeIcon, setActiveIcon] = useState(null);

  const handleIconClick = (iconName) => {
    setActiveIcon(iconName);
  };

  const icons = [
    { name: 'Discussion', icon: BellIcon },
    { name: 'Tableaux de bord', icon: GridIcon },
  ];

  return (
    <div style={{ backgroundColor: '#F3F4F6', minHeight: '100vh', padding: '2rem' }}>
      <div style={{ maxWidth: '64rem', margin: '0 auto' }}>
        <div style={{
          backgroundColor: 'white',
          borderRadius: '0.5rem',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          padding: '2rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem' }}>
            {icons.map((icon) => (
              <DashboardIcon 
                key={icon.name}
                icon={icon.icon} 
                label={icon.name} 
                isActive={activeIcon === icon.name}
                onClick={() => handleIconClick(icon.name)}
              />
            ))}
          </div>
          {activeIcon && (
            <div style={{ marginTop: '2rem', textAlign: 'center', fontSize: '1rem', color: '#4F46E5' }}>
              You clicked on {activeIcon}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;