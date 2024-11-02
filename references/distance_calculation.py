from geopy.distance import geodesic

def calculate_distance_chunk(df_chunk):
    df_chunk['distance'] = df_chunk.apply(lambda row: geodesic((row['lat'], row['long']), 
                                                               (row['merch_lat'], row['merch_long'])).km, axis=1)
    return df_chunk
