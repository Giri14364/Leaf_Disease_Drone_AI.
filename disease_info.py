# ============================================================
# CROP DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    # --------------------------------------------------------
    # APPLE
    # --------------------------------------------------------

    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple Scab",
        "symptoms": (
            "Dark olive or brown spots can appear on leaves and fruit. "
            "Severe infection may cause leaf drop."
        ),
        "management": (
            "Remove infected plant material, maintain good air circulation, "
            "and follow locally recommended fungicide practices."
        )
    },

    "Apple___Black_rot": {
        "crop": "Apple",
        "disease": "Black Rot",
        "symptoms": (
            "Brown or purple leaf spots may develop and can enlarge over time."
        ),
        "management": (
            "Remove infected leaves and fruit, improve orchard sanitation, "
            "and follow recommended disease-management practices."
        )
    },

    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "symptoms": (
            "Yellow-orange spots can appear on leaves and may develop "
            "into larger lesions."
        ),
        "management": (
            "Remove infected plant material and use appropriate disease "
            "management practices recommended for the growing region."
        )
    },

    "Apple___healthy": {
        "crop": "Apple",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular monitoring, proper irrigation, nutrition, "
            "and field sanitation."
        )
    },

    # --------------------------------------------------------
    # BLUEBERRY
    # --------------------------------------------------------

    "Blueberry___healthy": {
        "crop": "Blueberry",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular monitoring and maintain appropriate "
            "water, nutrition, and sanitation."
        )
    },

    # --------------------------------------------------------
    # CHERRY
    # --------------------------------------------------------

    "Cherry_(including_sour)___healthy": {
        "crop": "Cherry",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular monitoring and maintain good plant health."
        )
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "crop": "Cherry",
        "disease": "Powdery Mildew",
        "symptoms": (
            "White powder-like growth may appear on leaves and young plant tissue."
        ),
        "management": (
            "Improve air circulation, remove heavily affected material, "
            "and follow locally recommended fungicide practices."
        )
    },

    # --------------------------------------------------------
    # CORN
    # --------------------------------------------------------

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Corn",
        "disease": "Cercospora Leaf Spot / Gray Leaf Spot",
        "symptoms": (
            "Long gray, tan, or brown lesions may appear on leaves."
        ),
        "management": (
            "Use appropriate crop rotation and resistant varieties where "
            "available and follow local disease-management recommendations."
        )
    },

    "Corn_(maize)___Common_rust_": {
        "crop": "Corn",
        "disease": "Common Rust",
        "symptoms": (
            "Small reddish-brown rust-colored pustules may appear on leaves."
        ),
        "management": (
            "Monitor fields regularly and use resistant varieties where "
            "available."
        )
    },

    "Corn_(maize)___healthy": {
        "crop": "Corn",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular crop monitoring and maintain appropriate "
            "irrigation and nutrition."
        )
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn",
        "disease": "Northern Leaf Blight",
        "symptoms": (
            "Long gray-green to brown lesions can develop on corn leaves."
        ),
        "management": (
            "Use resistant varieties, appropriate crop rotation, and "
            "locally recommended disease-management practices."
        )
    },

    # --------------------------------------------------------
    # GRAPE
    # --------------------------------------------------------

    "Grape___Black_rot": {
        "crop": "Grape",
        "disease": "Black Rot",
        "symptoms": (
            "Brown or reddish leaf spots may develop and later become dark."
        ),
        "management": (
            "Remove infected plant material, maintain vineyard sanitation, "
            "and follow recommended disease-control practices."
        )
    },

    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape",
        "disease": "Esca (Black Measles)",
        "symptoms": (
            "Leaf discoloration and characteristic spotting may occur."
        ),
        "management": (
            "Remove affected plant material where appropriate and follow "
            "local vineyard disease-management recommendations."
        )
    },

    "Grape___healthy": {
        "crop": "Grape",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular monitoring and maintain good vineyard sanitation."
        )
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "symptoms": (
            "Leaf spots and blighted areas may develop on grape leaves."
        ),
        "management": (
            "Remove infected plant material, improve air circulation, "
            "and follow locally recommended disease-management practices."
        )
    },

    # --------------------------------------------------------
    # ORANGE
    # --------------------------------------------------------

    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "symptoms": (
            "Leaves may show uneven yellowing or blotchy mottling and "
            "overall plant decline may occur."
        ),
        "management": (
            "Monitor citrus plants carefully and follow local citrus "
            "greening management and vector-control recommendations."
        )
    },

    # --------------------------------------------------------
    # PEACH
    # --------------------------------------------------------

    "Peach___Bacterial_spot": {
        "crop": "Peach",
        "disease": "Bacterial Spot",
        "symptoms": (
            "Small dark spots may develop on leaves and fruit."
        ),
        "management": (
            "Maintain good orchard sanitation and follow locally "
            "recommended bacterial disease-management practices."
        )
    },

    "Peach___healthy": {
        "crop": "Peach",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular monitoring and maintain proper irrigation "
            "and nutrition."
        )
    },

    # --------------------------------------------------------
    # PEPPER
    # --------------------------------------------------------

    "Pepper,_bell___Bacterial_spot": {
        "crop": "Pepper",
        "disease": "Bacterial Spot",
        "symptoms": (
            "Small dark water-soaked or brown spots may appear on leaves."
        ),
        "management": (
            "Use healthy planting material, avoid unnecessary leaf wetness, "
            "and follow local bacterial disease-management recommendations."
        )
    },

    "Pepper,_bell___healthy": {
        "crop": "Pepper",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular monitoring and maintain appropriate "
            "irrigation and nutrition."
        )
    },

    # --------------------------------------------------------
    # POTATO
    # --------------------------------------------------------

    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "symptoms": (
            "Dark brown spots with concentric ring patterns may develop "
            "on potato leaves."
        ),
        "management": (
            "Remove infected plant material, avoid prolonged leaf wetness, "
            "and follow recommended fungicide practices."
        )
    },

    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular monitoring and maintain appropriate "
            "irrigation and nutrition."
        )
    },

    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "symptoms": (
            "Dark water-soaked lesions may develop on leaves and stems."
        ),
        "management": (
            "Remove severely infected material and follow local late-blight "
            "management recommendations."
        )
    },

    # --------------------------------------------------------
    # RASPBERRY
    # --------------------------------------------------------

    "Raspberry___healthy": {
        "crop": "Raspberry",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular monitoring and maintain good plant health."
        )
    },

    # --------------------------------------------------------
    # SOYBEAN
    # --------------------------------------------------------

    "Soybean___healthy": {
        "crop": "Soybean",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular field monitoring and maintain appropriate "
            "water and nutrient management."
        )
    },

    # --------------------------------------------------------
    # SQUASH
    # --------------------------------------------------------

    "Squash___Powdery_mildew": {
        "crop": "Squash",
        "disease": "Powdery Mildew",
        "symptoms": (
            "White powder-like fungal growth may appear on leaf surfaces."
        ),
        "management": (
            "Improve air circulation, remove heavily affected material, "
            "and follow locally recommended fungicide practices."
        )
    },

    # --------------------------------------------------------
    # STRAWBERRY
    # --------------------------------------------------------

    "Strawberry___healthy": {
        "crop": "Strawberry",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular monitoring and maintain good plant health."
        )
    },

    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry",
        "disease": "Leaf Scorch",
        "symptoms": (
            "Dark or reddish-brown spots may appear and expand on leaves."
        ),
        "management": (
            "Remove severely affected leaves, maintain good airflow, "
            "and follow local disease-management recommendations."
        )
    },

    # --------------------------------------------------------
    # TOMATO
    # --------------------------------------------------------

    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "symptoms": (
            "Small dark spots may appear on tomato leaves and fruit."
        ),
        "management": (
            "Use healthy planting material, avoid excessive leaf wetness, "
            "and follow locally recommended bacterial disease management."
        )
    },

    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "symptoms": (
            "Brown leaf spots with concentric ring patterns may develop."
        ),
        "management": (
            "Remove affected leaves, improve air circulation, avoid prolonged "
            "leaf wetness, and follow local disease-management recommendations."
        )
    },

    "Tomato___healthy": {
        "crop": "Tomato",
        "disease": "Healthy",
        "symptoms": "No obvious disease symptoms detected.",
        "management": (
            "Continue regular monitoring and maintain proper irrigation, "
            "nutrition, and field sanitation."
        )
    },

    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "symptoms": (
            "Dark water-soaked lesions can develop on leaves and stems."
        ),
        "management": (
            "Remove severely infected plant material and follow local "
            "late-blight management recommendations."
        )
    },

    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "symptoms": (
            "Yellowish spots may appear on upper leaf surfaces with "
            "mold-like growth underneath."
        ),
        "management": (
            "Improve ventilation, reduce prolonged leaf wetness, and "
            "follow locally recommended disease-management practices."
        )
    },

    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "symptoms": (
            "Small circular leaf spots with darker margins may develop."
        ),
        "management": (
            "Remove affected leaves, maintain field sanitation, and avoid "
            "unnecessary leaf wetness."
        )
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Spider Mites (Two-Spotted Spider Mite)",
        "symptoms": (
            "Fine webbing, stippling, yellowing, and leaf damage may occur."
        ),
        "management": (
            "Monitor the underside of leaves and follow integrated pest "
            "management practices appropriate for the crop."
        )
    },

    "Tomato___Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot",
        "symptoms": (
            "Circular brown lesions with target-like patterns may develop "
            "on leaves."
        ),
        "management": (
            "Remove affected plant material, improve airflow, and follow "
            "locally recommended disease-management practices."
        )
    },

    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "symptoms": (
            "Mottled or mosaic-like patterns and leaf distortion may occur."
        ),
        "management": (
            "Use clean planting material, sanitize tools and hands, and "
            "remove infected plants where appropriate."
        )
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "symptoms": (
            "Leaves may curl upward and show yellowing, with reduced plant growth."
        ),
        "management": (
            "Monitor and manage insect vectors according to local integrated "
            "pest management recommendations and remove severely infected plants."
        )
    }
}


# ============================================================
# GET DISEASE INFORMATION
# ============================================================

def get_disease_info(disease_class):
    """
    Return disease information for a PlantVillage class name.
    """

    return DISEASE_INFO.get(
        disease_class,
        {
            "crop": "Unknown",
            "disease": disease_class,
            "symptoms": "Disease information is not available yet.",
            "management": (
                "Please consult a qualified agricultural expert "
                "for field-level diagnosis and management."
            )
        }
    )