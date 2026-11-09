
-- 1. Dimension entreprise
CREATE TABLE dim_company (
    company_id SERIAL PRIMARY KEY,
    company_name VARCHAR(255) UNIQUE
);

-- 2. Dimension localisation
CREATE TABLE dim_location (
    location_id SERIAL PRIMARY KEY,
    city VARCHAR(255) UNIQUE
);

-- 3. Dimension contrat
CREATE TABLE dim_contract (
    contract_id SERIAL PRIMARY KEY,
    contract_type VARCHAR(100) UNIQUE
);

-- 4. Dimension secteur
CREATE TABLE dim_sector (
    sector_id SERIAL PRIMARY KEY,
    sector_name VARCHAR(255) UNIQUE
);

-- 5. Dimension expérience
CREATE TABLE dim_experience (
    experience_id SERIAL PRIMARY KEY,
    experience_level VARCHAR(255) UNIQUE
);

-- 6. Dimension date
CREATE TABLE dim_date (
    date_id SERIAL PRIMARY KEY,
    publication_date DATE UNIQUE,
    day INTEGER,
    month INTEGER,
    year INTEGER
);

-- 7. Dimension descriptive de l'offre
CREATE TABLE dim_offer (
    offer_id VARCHAR(100) PRIMARY KEY,
    title TEXT,
    description TEXT,
    url TEXT,
    remote VARCHAR(100),
    education_level VARCHAR(255)
);

CREATE TABLE fact_job_offers (
    offer_id VARCHAR(100) PRIMARY KEY,

    company_id INTEGER REFERENCES dim_company(company_id),
    location_id INTEGER REFERENCES dim_location(location_id),
    contract_id INTEGER REFERENCES dim_contract(contract_id),
    sector_id INTEGER REFERENCES dim_sector(sector_id),
    experience_id INTEGER REFERENCES dim_experience(experience_id),
    date_id INTEGER REFERENCES dim_date(date_id),

    offer_count INTEGER DEFAULT 1,

    CONSTRAINT fk_offer
        FOREIGN KEY (offer_id)
        REFERENCES dim_offer(offer_id)
);
