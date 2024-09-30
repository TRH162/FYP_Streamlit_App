import streamlit as st 
import gdown 
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Google Drive URL to the pickle file
url = 'https://drive.google.com/uc?id=1jZ-V5bhfxU5UCNwBlwM6KdODKyHBqNK1'
output = 'my_pickle_file.pkl'
 
# Download the pickle file from Google Drive
@st.cache(allow_output_mutation=True)
def download_model():
    gdown.download(url, output, quiet=False)
    with open(output, 'rb') as file:
        loaded_objects = pickle.load(file)
    return loaded_objects
 
# Load the necessary objects (
objects = download_model()
model = objects['model']
label_encoder = objects['label_encoder']
customer_category_mapping = objects['customer_category_mapping']
day_sin_transformer = objects['day_sin_transformer']
day_cos_transformer = objects['day_cos_transformer']

# Streamlit app title
st.title('PREDICTION OF ACCIDENT SEVERITY')
st.write('This app predicts road accident severity (Slight/Serious) in The United Kingdom')

# Input features
# Junction detail
junction_detail = st.selectbox(
    'Junction_Detail',
    ['Crossroads', 'Mini-roundabout', 'More than 4 arms (not roundabout)', 'Not at junction or within 20 metres', 'Other junction', 
     'Private drive or entrance', 'Roundabout', 'Slip road', 'T or staggered junction']
)

# Latitude
latitude = st.slider('Latitude', 49.914488, 60.598055, 57.187246)

# Longitude
longitude = st.slider('Longitude', -7.516225, 1.759398,-2.168717)

# District Area
local_authority_district = st.selectbox(
    'Local_Authority_(District)',
    ['Aberdeen City', 'Aberdeenshire', 'Adur', 'Allerdale', 'Alnwick', 'Amber Valley', 'Angus', 'Argyll and Bute', 'Arun', 'Ashfield',
     'Ashford', 'Aylesbury Vale', 'Babergh', 'Barking and Dagenham', 'Barnet', 'Barnsley', 'Barrow-in-Furness', 'Basildon', 'Basingstoke and Deane', 'Bassetlaw',
     'Bath and North East Somerset', 'Bedford', 'Berwick-upon-Tweed', 'Bexley', 'Birmingham', 'Blaby', 'Blackburn with Darwen', 'Blackpool', 'Blaenau Gwent', 'Blyth Valley',
     'Bolsover', 'Bolton', 'Boston', 'Bournemouth', 'Bracknell Forest', 'Bradford', 'Braintree', 'Breckland', 'Brent', 'Brentwood',
     'Bridgend', 'Bridgnorth', 'Brighton and Hove', 'Bristol, City of', 'Broadland', 'Bromley', 'Bromsgrove', 'Broxbourne', 'Broxtowe', 'Burnley',
     'Bury', 'Caerphilly', 'Calderdale', 'Cambridge', 'Camden', 'Cannock Chase', 'Canterbury', 'Caradon', 'Cardiff', 'Carlisle',
     'Carmarthenshire', 'Carrick', 'Castle Morpeth', 'Castle Point', 'Central Bedfordshire', 'Ceredigion', 'Charnwood', 'Chelmsford', 'Cheltenham', 'Cherwell',
     'Cheshire East', 'Cheshire West and Chester', 'Chester', 'Chester-le-Street', 'Chesterfield', 'Chichester', 'Chiltern', 'Chorley', 'Christchurch', 'City of London',
     'Clackmannanshire', 'Colchester', 'Congleton', 'Conwy', 'Copeland', 'Corby', 'Cornwall', 'Cotswold', 'County Durham', 'Coventry',
     'Craven', 'Crawley', 'Crewe and Nantwich', 'Croydon', 'Dacorum', 'Darlington', 'Dartford', 'Daventry', 'Denbighshire', 'Derby',
     'Derbyshire Dales', 'Derwentside', 'Doncaster', 'Dover', 'Dudley', 'Dumfries and Galloway', 'Dundee City', 'Durham', 'Ealing', 'Easington',
     'East Ayrshire', 'East Cambridgeshire', 'East Devon', 'East Dorset', 'East Dunbartonshire', 'East Hampshire', 'East Hertfordshire', 'East Lindsey', 'East Lothian', 'East Northamptonshire',
     'East Renfrewshire', 'East Riding of Yorkshire', 'East Staffordshire', 'Eastbourne', 'Eastleigh', 'Eden', 'Edinburgh, City of', 'Ellesmere Port and Neston', 'Elmbridge', 'Enfield',
     'Epping Forest', 'Epsom and Ewell', 'Erewash', 'Exeter', 'Falkirk', 'Fareham', 'Fenland', 'Fife', 'Flintshire', 'Forest Heath',
     'Forest of Dean', 'Fylde', 'Gateshead', 'Gedling', 'Glasgow City', 'Gloucester', 'Gosport', 'Gravesham', 'Great Yarmouth', 'Greenwich',
     'Guildford', 'Gwynedd', 'Hackney', 'Halton', 'Hambleton', 'Hammersmith and Fulham', 'Harborough', 'Haringey', 'Harlow', 'Harrogate',
     'Harrow', 'Hart', 'Hartlepool', 'Hastings', 'Havant', 'Havering', 'Herefordshire, County of', 'Hertsmere', 'High Peak', 'Highland',
     'Hillingdon', 'Hinckley and Bosworth', 'Horsham', 'Hounslow', 'Huntingdonshire', 'Hyndburn', 'Inverclyde', 'Ipswich', 'Isle of Anglesey', 'Isle of Wight',
     'Islington', 'Kennet', 'Kensington and Chelsea', 'Kerrier', 'Kettering', 'Kings Lynn and West Norfolk', 'Kingston upon Hull, City of', 'Kingston upon Thames', 'Kirklees', 'Knowsley',
     'Lambeth', 'Lancaster', 'Leeds', 'Leicester', 'Lewes', 'Lewisham', 'Lichfield', 'Lincoln', 'Liverpool', 'London Airport (Heathrow)',
     'Luton', 'Macclesfield', 'Maidstone', 'Maldon', 'Malvern Hills', 'Manchester', 'Mansfield', 'Medway', 'Melton', 'Mendip',
     'Merthyr Tydfil', 'Merton', 'Mid Bedfordshire', 'Mid Devon', 'Mid Suffolk', 'Mid Sussex', 'Middlesbrough', 'Midlothian', 'Milton Keynes', 'Mole Valley',
     'Monmouthshire', 'Moray', 'Neath Port Talbot', 'New Forest', 'Newark and Sherwood', 'Newcastle upon Tyne', 'Newcastle-under-Lyme', 'Newham', 'Newport', 'North Ayrshire',
     'North Cornwall', 'North Devon', 'North Dorset', 'North East Derbyshire', 'North East Lincolnshire', 'North Hertfordshire', 'North Kesteven', 'North Lanarkshire', 'North Larkshire', 'North Lincolnshire',
     'North Norfolk', 'North Shropshire', 'North Somerset', 'North Tyneside', 'North Warwickshire', 'North West Leicestershire', 'North Wiltshire', 'Northampton', 'Northumberland', 'Norwich',
     'Nottingham', 'Nuneaton and Bedworth', 'Oadby and Wigston', 'Oldham', 'Orkney Islands', 'Oswestry', 'Oxford', 'Pembrokeshire', 'Pendle', 'Penwith',
     'Perth and Kinross', 'Peterborough', 'Plymouth', 'Poole', 'Portsmouth', 'Powys', 'Preston', 'Purbeck', 'Reading', 'Redbridge',
     'Redcar and Cleveland', 'Redditch', 'Reigate and Banstead', 'Renfrewshire', 'Restormel', 'Rhondda, Cynon, Taff', 'Ribble Valley', 'Richmond upon Thames', 'Richmondshire', 'Rochdale',
     'Rochford', 'Rossendale', 'Rother', 'Rotherham', 'Rugby', 'Runnymede', 'Rushcliffe', 'Rushmoor', 'Rutland', 'Ryedale',
     'Salford', 'Salisbury', 'Sandwell', 'Scarborough', 'Scottish Borders', 'Sedgefield', 'Sedgemoor', 'Sefton', 'Selby', 'Sevenoaks',
     'Sheffield', 'Shepway', 'Shetland Islands', 'Shrewsbury and Atcham', 'Shropshire', 'Slough', 'Solihull', 'South Ayrshire', 'South Bedfordshire', 'South Bucks',
     'South Cambridgeshire', 'South Derbyshire', 'South Gloucestershire', 'South Hams', 'South Holland', 'South Kesteven', 'South Lakeland', 'South Lanarkshire', 'South Larkshire', 'South Norfolk',
     'South Northamptonshire', 'South Oxfordshire', 'South Ribble', 'South Shropshire', 'South Somerset', 'South Staffordshire', 'South Tyneside', 'Southampton', 'Southend-on-Sea', 'Southwark',
     'Spelthorne', 'St. Albans', 'St. Edmundsbury', 'St. Helens', 'Stafford', 'Staffordshire Moorlands', 'Stevenage', 'Stirling', 'Stockport', 'Stockton-on-Tees',
     'Stoke-on-Trent', 'Stratford-upon-Avon', 'Stroud', 'Suffolk Coastal', 'Sunderland', 'Surrey Heath', 'Sutton', 'Swale', 'Swansea', 'Swindon',
     'Tameside', 'Tamworth', 'Tandridge', 'Taunton Deane', 'Teesdale', 'Teignbridge', 'Telford and Wrekin', 'Tendring', 'Test Valley', 'Tewkesbury',
     'Thanet', 'The Vale of Glamorgan', 'Three Rivers', 'Thurrock', 'Tonbridge and Malling', 'Torbay', 'Torfaen', 'Torridge', 'Tower Hamlets', 'Trafford',
     'Tunbridge Wells', 'Tynedale', 'Uttlesford', 'Vale Royal', 'Vale of White Horse', 'Wakefield', 'Walsall', 'Waltham Forest', 'Wandsworth', 'Wansbeck',
     'Warrington', 'Warwick', 'Watford', 'Waveney', 'Waverley', 'Wealden', 'Wear Valley', 'Wellingborough', 'Welwyn Hatfield', 'West Berkshire',
     'West Devon', 'West Dorset', 'West Dunbartonshire', 'West Lancashire', 'West Lindsey', 'West Lothian', 'West Oxfordshire', 'West Somerset', 'West Wiltshire', 'Western Isles',
     'Westminster', 'Weymouth and Portland', 'Wigan', 'Wiltshire', 'Winchester', 'Windsor and Maidenhead', 'Wirral', 'Woking', 'Wokingham', 'Wolverhampton',
     'Worcester', 'Worthing', 'Wrexham', 'Wychavon', 'Wycombe', 'Wyre', 'Wyre Forest', 'York']
)



# Light conditions
light_conditions = st.radio('Light_Conditions', ['Darkness', 'Daylight'])

# Number of casualties
number_of_casualties = st.slider('Number_of_Casualties', 1, 10, 1)

# Number of vehicles
number_of_vehicles = st.slider('Number_of_Vehicle', 1, 8, 1)

# Road surface conditions
road_surface_conditions = st.selectbox(
    'Road_Surface_Conditions',
    ['Dry', 'Flood over 3cm. deep', 'Frost or ice', 'Snow', 'Wet or damp']
)

# Road type
road_type = st.selectbox(
    'Road_Type',
    ['Dual carriageway', 'One way street', 'Roundabout', 'Single carriageway', 'Slip road']
)

# Urban or Rural area
urban_or_rural_area = st.radio('Urban_or_Rural_area', ['Urban', 'Rural'])

# Wehicle type
vehicle_type = st.selectbox(
    'Vehicle_Type',
    ['Agricultural vehicle', 'Bus or coach (17 or more pass seats)', 'Car', 'Goods 7.5 tonnes mgw and over', 'Minibus(8 - 16 passenger seats)', 'Motorcycle 125cc and under', 'Motorcycle 50cc and under',
     'Motorcycle over 125cc and up to 500cc', 'Motorcycle over 500cc', 'Other vehicle', 'Pedal cycle', 'Ridden horse', 'Taxi/Private hire car', 
     'Van / Goods 3.5 tonnes mgw or under', 'Goods over 3.5t. and under 7.5t']
)

# Day of week
day_of_week = st.selectbox(
    'Day_of_Week',
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

# Preprocessing the input
def preprocess_inputs():
    # Apply cyclical transformation for Day_of_Week
    day_of_week_encoded = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day_of_week)
    day_sin, day_cos = np.sin(2 * np.pi * day_of_week_encoded / 7), np.cos(2 * np.pi * day_of_week_encoded / 7)
    
    # Map categorical features
    junction_detail_mapping = {'Crossroads':0, 
                               'Mini-roundabout':1, 
                               'More than 4 arms (not roundabout)':2, 
                               'Not at junction or within 20 metres':3, 
                               'Other junction':4, 
                               'Private drive or entrance':5, 
                               'Roundabout':6, 
                               'Slip road':7, 
                               'T or staggered junction':8}
    
    light_conditions_mapping = {'Darkness':0, 'Daylight':1}
    
    road_surface_conditions_mapping = {'Dry':0, 'Flood over 3cm. deep':1, 'Frost or ice':2, 'Snow':3, 'Wet or damp':4}
    
    road_type_mapping = {'Dual carriageway':0, 'One way street':1, 'Roundabout':2, 'Single carriageway':3, 'Slip road':4}
    
    urban_or_rural_area_mapping = {'Urban':0, 'Rural':1}
    
    vehicle_type_mapping = {'Agricultural vehicle':0, 'Bus or coach (17 or more pass seats)':1, 'Car':2, 'Goods 7.5 tonnes mgw and over':3, 'Goods over 3.5t. and under 7.5t':4, 'Minibus(8 - 16 passenger seats)':5, 
                            'Motorcycle 125cc and under':6, 'Motorcycle 50cc and under':7,'Motorcycle over 125cc and up to 500cc':8, 'Motorcycle over 500cc':9, 'Other vehicle':10, 
                            'Pedal cycle':11, 'Ridden horse':12, 'Taxi/Private hire car':13, 'Van / Goods 3.5 tonnes mgw or under':14}
    
    local_authority_mapping = {
        'Aberdeen City':0, 'Aberdeenshire':1, 'Adur':2, 'Allerdale':3, 'Alnwick':4, 'Amber Valley':5, 'Angus':6, 'Argyll and Bute':7, 'Arun':8, 'Ashfield':9,
        'Ashford':10, 'Aylesbury Vale':11, 'Babergh':12, 'Barking and Dagenham':13, 'Barnet':14, 'Barnsley':15, 'Barrow-in-Furness':16, 'Basildon':17, 'Basingstoke and Deane':18, 'Bassetlaw':19,
        'Bath and North East Somerset':20, 'Bedford':21, 'Berwick-upon-Tweed':22, 'Bexley':23, 'Birmingham':24, 'Blaby':25, 'Blackburn with Darwen':26, 'Blackpool':27, 'Blaenau Gwent':28, 'Blyth Valley':29,
        'Bolsover':30, 'Bolton':31, 'Boston':32, 'Bournemouth':33, 'Bracknell Forest':34, 'Bradford':35, 'Braintree':36, 'Breckland':37, 'Brent':38, 'Brentwood':39,
        'Bridgend':40, 'Bridgnorth':41, 'Brighton and Hove':42, 'Bristol, City of':43, 'Broadland':44, 'Bromley':45, 'Bromsgrove':46, 'Broxbourne':47, 'Broxtowe':48, 'Burnley':49,
        'Bury':50, 'Caerphilly':51, 'Calderdale':52, 'Cambridge':53, 'Camden':54, 'Cannock Chase':55, 'Canterbury':56, 'Caradon':57, 'Cardiff':58, 'Carlisle':59,
        'Carmarthenshire':60, 'Carrick':61, 'Castle Morpeth':62, 'Castle Point':63, 'Central Bedfordshire':64, 'Ceredigion':65, 'Charnwood':66, 'Chelmsford':67, 'Cheltenham':68, 'Cherwell':69,
        'Cheshire East':70, 'Cheshire West and Chester':71, 'Chester':72, 'Chester-le-Street':73, 'Chesterfield':74, 'Chichester':75, 'Chiltern':76, 'Chorley':77, 'Christchurch':78, 'City of London':79,
        'Clackmannanshire':80, 'Colchester':81, 'Congleton':82, 'Conwy':83, 'Copeland':84, 'Corby':85, 'Cornwall':86, 'Cotswold':87, 'County Durham':88, 'Coventry':89,
        'Craven':90, 'Crawley':91, 'Crewe and Nantwich':92, 'Croydon':93, 'Dacorum':94, 'Darlington':95, 'Dartford':96, 'Daventry':97, 'Denbighshire':98, 'Derby':99,
        'Derbyshire Dales':100, 'Derwentside':101, 'Doncaster':102, 'Dover':103, 'Dudley':104, 'Dumfries and Galloway':105, 'Dundee City':106, 'Durham':107, 'Ealing':108, 'Easington':109,
        'East Ayrshire':110, 'East Cambridgeshire':111, 'East Devon':112, 'East Dorset':113, 'East Dunbartonshire':114, 'East Hampshire':115, 'East Hertfordshire':116, 'East Lindsey':117, 'East Lothian':118, 'East Northamptonshire':119,
        'East Renfrewshire':120, 'East Riding of Yorkshire':121, 'East Staffordshire':122, 'Eastbourne':123, 'Eastleigh':124, 'Eden':125, 'Edinburgh, City of':126, 'Ellesmere Port and Neston':127, 'Elmbridge':128, 'Enfield':129,
        'Epping Forest':130, 'Epsom and Ewell':131, 'Erewash':132, 'Exeter':133, 'Falkirk':134, 'Fareham':135, 'Fenland':136, 'Fife':137, 'Flintshire':138, 'Forest Heath':139,
        'Forest of Dean':140, 'Fylde':141, 'Gateshead':142, 'Gedling':143, 'Glasgow City':144, 'Gloucester':145, 'Gosport':146, 'Gravesham':147, 'Great Yarmouth':148, 'Greenwich':149,
        'Guildford':150, 'Gwynedd':151, 'Hackney':152, 'Halton':153, 'Hambleton':154, 'Hammersmith and Fulham':155, 'Harborough':156, 'Haringey':157, 'Harlow':158, 'Harrogate':159,
        'Harrow':160, 'Hart':161, 'Hartlepool':162, 'Hastings':163, 'Havant':164, 'Havering':165, 'Herefordshire, County of':166, 'Hertsmere':167, 'High Peak':168, 'Highland':169,
        'Hillingdon':170, 'Hinckley and Bosworth':171, 'Horsham':172, 'Hounslow':173, 'Huntingdonshire':174, 'Hyndburn':175, 'Inverclyde':176, 'Ipswich':177, 'Isle of Anglesey':178, 'Isle of Wight':179,
        'Islington':180, 'Kennet':181, 'Kensington and Chelsea':182, 'Kerrier':183, 'Kettering':184, 'Kings Lynn and West Norfolk':185, 'Kingston upon Hull, City of':186, 'Kingston upon Thames':187, 'Kirklees':188, 'Knowsley':189,
        'Lambeth':190, 'Lancaster':191, 'Leeds':192, 'Leicester':193, 'Lewes':194, 'Lewisham':195, 'Lichfield':196, 'Lincoln':197, 'Liverpool':198, 'London Airport (Heathrow)':199,
        'Luton':200, 'Macclesfield':201, 'Maidstone':202, 'Maldon':203, 'Malvern Hills':204, 'Manchester':205, 'Mansfield':206, 'Medway':207, 'Melton':208, 'Mendip':209,
        'Merthyr Tydfil':210, 'Merton':211, 'Mid Bedfordshire':212, 'Mid Devon':213, 'Mid Suffolk':214, 'Mid Sussex':215, 'Middlesbrough':216, 'Midlothian':217, 'Milton Keynes':218, 'Mole Valley':219,
        'Monmouthshire':220, 'Moray':221, 'Neath Port Talbot':222, 'New Forest':223, 'Newark and Sherwood':224, 'Newcastle upon Tyne':225, 'Newcastle-under-Lyme':226, 'Newham':227, 'Newport':228, 'North Ayrshire':229,
        'North Cornwall':230, 'North Devon':231, 'North Dorset':232, 'North East Derbyshire':233, 'North East Lincolnshire':234, 'North Hertfordshire':235, 'North Kesteven':236, 'North Lanarkshire':237, 'North Larkshire':238, 'North Lincolnshire':239,
        'North Norfolk':240, 'North Shropshire':241, 'North Somerset':242, 'North Tyneside':243, 'North Warwickshire':244, 'North West Leicestershire':245, 'North Wiltshire':246, 'Northampton':247, 'Northumberland':248, 'Norwich':249,
        'Nottingham':250, 'Nuneaton and Bedworth':251, 'Oadby and Wigston':252, 'Oldham':253, 'Orkney Islands':254, 'Oswestry':255, 'Oxford':256, 'Pembrokeshire':257, 'Pendle':258, 'Penwith':259,
        'Perth and Kinross':260, 'Peterborough':261, 'Plymouth':262, 'Poole':263, 'Portsmouth':264, 'Powys':265, 'Preston':266, 'Purbeck':267, 'Reading':268, 'Redbridge':269,
        'Redcar and Cleveland':270, 'Redditch':271, 'Reigate and Banstead':272, 'Renfrewshire':273, 'Restormel':274, 'Rhondda, Cynon, Taff':275, 'Ribble Valley':276, 'Richmond upon Thames':277, 'Richmondshire':278, 'Rochdale':279,
        'Rochford':280, 'Rossendale':281, 'Rother':282, 'Rotherham':283, 'Rugby':284, 'Runnymede':285, 'Rushcliffe':286, 'Rushmoor':287, 'Rutland':288, 'Ryedale':289,
        'Salford':290, 'Salisbury':291, 'Sandwell':292, 'Scarborough':293, 'Scottish Borders':294, 'Sedgefield':295, 'Sedgemoor':296, 'Sefton':297, 'Selby':298, 'Sevenoaks':299,
        'Sheffield':300, 'Shepway':301, 'Shetland Islands':302, 'Shrewsbury and Atcham':303, 'Shropshire':304, 'Slough':305, 'Solihull':306, 'South Ayrshire':307, 'South Bedfordshire':308, 'South Bucks':309,
        'South Cambridgeshire':310, 'South Derbyshire':311, 'South Gloucestershire':312, 'South Hams':313, 'South Holland':314, 'South Kesteven':315, 'South Lakeland':316, 'South Lanarkshire':317, 'South Larkshire':318, 'South Norfolk':319,
        'South Northamptonshire':320, 'South Oxfordshire':321, 'South Ribble':322, 'South Shropshire':323, 'South Somerset':324, 'South Staffordshire':325, 'South Tyneside':326, 'Southampton':327, 'Southend-on-Sea':328, 'Southwark':329,
        'Spelthorne':330, 'St. Albans':331, 'St. Edmundsbury':332, 'St. Helens':333, 'Stafford':334, 'Staffordshire Moorlands':335, 'Stevenage':336, 'Stirling':337, 'Stockport':338, 'Stockton-on-Tees':339,
        'Stoke-on-Trent':340, 'Stratford-upon-Avon':341, 'Stroud':342, 'Suffolk Coastal':343, 'Sunderland':344, 'Surrey Heath':345, 'Sutton':346, 'Swale':347, 'Swansea':348, 'Swindon':349,
        'Tameside':350, 'Tamworth':351, 'Tandridge':352, 'Taunton Deane':353, 'Teesdale':354, 'Teignbridge':355, 'Telford and Wrekin':356, 'Tendring':357, 'Test Valley':358, 'Tewkesbury':359,
        'Thanet':360, 'The Vale of Glamorgan':361, 'Three Rivers':362, 'Thurrock':363, 'Tonbridge and Malling':364, 'Torbay':365, 'Torfaen':366, 'Torridge':367, 'Tower Hamlets':368, 'Trafford':369,
        'Tunbridge Wells':370, 'Tynedale':371, 'Uttlesford':372, 'Vale Royal':373, 'Vale of White Horse':374, 'Wakefield':375, 'Walsall':376, 'Waltham Forest':377, 'Wandsworth':378, 'Wansbeck':379,
        'Warrington':380, 'Warwick':381, 'Watford':382, 'Waveney':383, 'Waverley':384, 'Wealden':385, 'Wear Valley':386, 'Wellingborough':387, 'Welwyn Hatfield':388, 'West Berkshire':389,
        'West Devon':390, 'West Dorset':391, 'West Dunbartonshire':392, 'West Lancashire':393, 'West Lindsey':394, 'West Lothian':395, 'West Oxfordshire':396, 'West Somerset':397, 'West Wiltshire':398, 'Western Isles':399,
        'Westminster':400, 'Weymouth and Portland':401, 'Wigan':402, 'Wiltshire':403, 'Winchester':404, 'Windsor and Maidenhead':405, 'Wirral':406, 'Woking':407, 'Wokingham':408, 'Wolverhampton':409,
        'Worcester':410, 'Worthing':411, 'Wrexham':412, 'Wychavon':413, 'Wycombe':414, 'Wyre':415, 'Wyre Forest':416, 'York':417
    }
    
    junction_detail_selected = junction_detail_mapping[junction_detail]
    light_conditions_selected = light_conditions_mapping[light_conditions]
    road_surface_conditions_selected = road_surface_conditions_mapping[road_surface_conditions]
    road_type_selected = road_type_mapping[road_type]
    urban_or_rural_area_selected = urban_or_rural_area_mapping[urban_or_rural_area]
    vehicle_type_selected = vehicle_type_mapping[vehicle_type]
    local_authority_selected = local_authority_mapping[local_authority_district]
    
    # Combine all features into a single array
    input_features = np.array([day_sin, day_cos, latitude, longitude, number_of_vehicles, number_of_casualties, 
                               junction_detail_selected, local_authority_selected, light_conditions_selected, 
                               road_surface_conditions_selected, road_type_selected, urban_or_rural_area_selected, 
                               vehicle_type_selected]).reshape(1, -1)
    
    return input_features


# Predict and display result
if st.button("Predict Severity"):
    # Preprocess the inputs
    features = preprocess_inputs()

    # Get the predicted probabilities for both classes
    probabilities = model.predict_proba(features)
    probability_serious = probabilities[0][1]  # Probability of class 1 (Serious)
    probability_slight = probabilities[0][0]   # Probability of class 0 (Slight)

    # Display prediction and both probabilities
    st.write(f"Probability of **Serious**: {probability_serious:.2f}")
    st.write(f"Probability of **Slight**: {probability_slight:.2f}")
 
    
    # Apply custom logic or threshold to classify severity
    if probability_serious >= 0.55:  # Default threshold of 0.55
        st.markdown(f'The predicted severity is <strong>The predicted severity is</strong> <span style="color:red;"><strong>Serious</strong></span>.', unsafe_allow_html=True)
    else:
        st.markdown(f'The predicted severity is <strong>The predicted severity is</strong> <span style="color:green;"><strong>Slight</strong></span>.', unsafe_allow_html=True)
        
        

