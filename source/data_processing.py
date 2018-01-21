from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer

CATEGORIES = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
              'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1',
              'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
              'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond',
              'YrSold', 'MoSold', 'SaleCondition', 'SaleType', 'LotConfig', 'Utilities', 'LandContour',
              'MSZoning', 'GarageType', 'Electrical', 'Heating', 'HeatingQC', 'Foundation')


def remove_null_values(data_set):
	data_set_copy = data_set.copy()

	# Drop columns where large number of values are null
	columns_to_drop = ['PoolQC', 'Fence', 'MiscFeature', 'Alley']
	data_set_copy = data_set_copy.drop(columns_to_drop, axis=1)

	imputer = Imputer()
	data_set_copy[['LotFrontage']] = imputer.fit_transform(data_set_copy[['LotFrontage']])

	data_set_copy[['MasVnrType']] = data_set_copy[['MasVnrType']].fillna(value='None')
	data_set_copy[['MasVnrArea']] = data_set_copy[['MasVnrArea']].fillna(value=0)

	data_set_copy[['BsmtQual']] = data_set_copy[['BsmtQual']].fillna(value='None')
	data_set_copy[['BsmtCond']] = data_set_copy[['BsmtCond']].fillna(value=0)

	data_set_copy[['BsmtExposure']] = data_set_copy[['BsmtExposure']].fillna(value='NA')
	data_set_copy[['BsmtFinType1']] = data_set_copy[['BsmtFinType1']].fillna(value='NA')
	data_set_copy[['BsmtFinType2']] = data_set_copy[['BsmtFinType2']].fillna(value='NA')

	# Only 2 missing values, drop it
	data_set_copy = data_set_copy.dropna(subset=['Electrical'], how='all')

	data_set_copy[['FireplaceQu']] = data_set_copy[['FireplaceQu']].fillna(value='NA')

	data_set_copy[['GarageType']] = data_set_copy[['GarageType']].fillna(value='NA')
	data_set_copy[['GarageFinish']] = data_set_copy[['GarageFinish']].fillna(value='NA')
	data_set_copy[['GarageQual']] = data_set_copy[['GarageQual']].fillna(value='NA')
	data_set_copy[['GarageCond']] = data_set_copy[['GarageCond']].fillna(value='NA')
	data_set_copy[['GarageYrBlt']] = data_set_copy[['GarageYrBlt']].fillna(value=0)

	data_set_copy['MSSubClass'] = data_set_copy['MSSubClass'].apply(str)
	data_set_copy['OverallCond'] = data_set_copy['OverallCond'].astype(str)
	data_set_copy['YrSold'] = data_set_copy['YrSold'].astype(str)
	data_set_copy['MoSold'] = data_set_copy['MoSold'].astype(str)

	return data_set_copy


def encode_data(data_set):
	labeler = LabelEncoder()
	data_set_copy = data_set.copy()
	for c in CATEGORIES:
		labeler.fit(list(data_set_copy[c].values))
		data_set_copy[c] = labeler.transform(list(data_set_copy[c].values))
	return data_set_copy


def scale_data(data_set):
	data_set_copy = data_set.copy()
	ks = StandardScaler().fit(data_set_copy)
	scaled_data = ks.transform(data_set_copy)
	return ks, scaled_data
