import React from 'react';
import { Link } from 'react-router-dom';
import siteConfig from '../config/siteConfig';

const Header = () => {
  return (
    <header className="site-header">
      <div className="container">
        <h1><Link to="/">{siteConfig.title}</Link></h1>
        <nav>
          <ul>
            <li><Link to="/services">Services</Link></li>
            <li><Link to="/solutions">Solutions</Link></li>
            <li><Link to="/about">À propos</Link></li>
            <li><Link to="/contact">Contact</Link></li>
            <li><Link to="/blog">Blog</Link></li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;