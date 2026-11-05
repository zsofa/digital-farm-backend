CREATE TABLE app.farm (
    id BIGSERIAL PRIMARY KEY,

    name VARCHAR(150) NOT NULL,
    county_name VARCHAR(100),

    activity_type VARCHAR(30) NOT NULL
        CHECK (
            activity_type IN (
                'crop_production',
                'livestock',
                'mixed'
            )
        ),

    legal_form VARCHAR(30)
        CHECK (
            legal_form IS NULL
            OR legal_form IN (
                'individual',
                'family_farm',
                'company',
                'other'
            )
        ),

    description TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);