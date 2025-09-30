import React from 'react';
import './BrandLogo.css';
import { appConfig } from '@/config';

const BrandLogo: React.FC = () => {
  return (
    <div className="brand-logo">
      <h1>{appConfig.title}</h1>
      <p>{appConfig.subtitle}</p>
    </div>
  );
};

export default BrandLogo;
