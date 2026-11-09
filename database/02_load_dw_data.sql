INSERT INTO dim_company (company_name)
SELECT DISTINCT TRIM(company)
FROM staging_job_offers
WHERE company IS NOT NULL AND TRIM(company) <> ''
ON CONFLICT (company_name) DO NOTHING;

INSERT INTO dim_location (city)
SELECT DISTINCT TRIM(location)
FROM staging_job_offers
WHERE location IS NOT NULL AND TRIM(location) <> ''
ON CONFLICT (city) DO NOTHING;

INSERT INTO dim_contract (contract_type)
SELECT DISTINCT TRIM(contract_type)
FROM staging_job_offers
WHERE contract_type IS NOT NULL AND TRIM(contract_type) <> ''
ON CONFLICT (contract_type) DO NOTHING;

INSERT INTO dim_sector (sector_name)
SELECT DISTINCT TRIM(sector)
FROM staging_job_offers
WHERE sector IS NOT NULL AND TRIM(sector) <> ''
ON CONFLICT (sector_name) DO NOTHING;

INSERT INTO dim_experience (experience_level)
SELECT DISTINCT TRIM(experience)
FROM staging_job_offers
WHERE experience IS NOT NULL AND TRIM(experience) <> ''
ON CONFLICT (experience_level) DO NOTHING;

INSERT INTO dim_offer (
    offer_id, title, description,
    url, remote, education_level
)
SELECT
    offer_id, title, description,
    url, remote, education_level
FROM staging_job_offers
ON CONFLICT (offer_id) DO NOTHING;

SELECT publication_date
FROM staging_job_offers
LIMIT 15;

INSERT INTO dim_date (
    publication_date,
    day,
    month,
    year
)
SELECT DISTINCT
    TO_DATE(TRIM(publication_date), 'DD/MM/YYYY'),
    EXTRACT(DAY FROM TO_DATE(TRIM(publication_date), 'DD/MM/YYYY')),
    EXTRACT(MONTH FROM TO_DATE(TRIM(publication_date), 'DD/MM/YYYY')),
    EXTRACT(YEAR FROM TO_DATE(TRIM(publication_date), 'DD/MM/YYYY'))
FROM staging_job_offers
WHERE publication_date IS NOT NULL
  AND TRIM(publication_date) <> ''
ON CONFLICT (publication_date) DO NOTHING;

SELECT *
FROM dim_date
ORDER BY publication_date
LIMIT 15;

INSERT INTO fact_job_offers (
    offer_id,
    company_id,
    location_id,
    contract_id,
    sector_id,
    experience_id,
    date_id,
    offer_count
)
SELECT
    s.offer_id,
    c.company_id,
    l.location_id,
    ct.contract_id,
    sec.sector_id,
    e.experience_id,
    d.date_id,
    1
FROM staging_job_offers s
LEFT JOIN dim_company c
    ON TRIM(s.company) = c.company_name
LEFT JOIN dim_location l
    ON TRIM(s.location) = l.city
LEFT JOIN dim_contract ct
    ON TRIM(s.contract_type) = ct.contract_type
LEFT JOIN dim_sector sec
    ON TRIM(s.sector) = sec.sector_name
LEFT JOIN dim_experience e
    ON TRIM(s.experience) = e.experience_level
LEFT JOIN dim_date d
    ON TO_DATE(TRIM(s.publication_date), 'DD/MM/YYYY')
       = d.publication_date
ON CONFLICT (offer_id) DO NOTHING;

SELECT *
FROM fact_job_offers
LIMIT 10;

SELECT COUNT(*)
FROM fact_job_offers;

SELECT
    COUNT(*) AS total_offers,
    COUNT(*) FILTER (WHERE company_id IS NULL) AS missing_company,
    COUNT(*) FILTER (WHERE location_id IS NULL) AS missing_location,
    COUNT(*) FILTER (WHERE contract_id IS NULL) AS missing_contract,
    COUNT(*) FILTER (WHERE sector_id IS NULL) AS missing_sector,
    COUNT(*) FILTER (WHERE experience_id IS NULL) AS missing_experience,
    COUNT(*) FILTER (WHERE date_id IS NULL) AS missing_date
FROM fact_job_offers;