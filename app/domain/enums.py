from enum import Enum


class SportType(str, Enum):
    BADMINTON = "BADMINTON"
    FOOTBALL = "FOOTBALL"
    TENNIS = "TENNIS"
    PICKLEBALL = "PICKLEBALL"
    BASKETBALL = "BASKETBALL"
    VOLLEYBALL = "VOLLEYBALL"


class SkillLevel(str, Enum):
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"
    PRO = "PRO"


class ParticipantType(str, Enum):
    PLAYER = "PLAYER"
    CLUB = "CLUB"


class AcceptanceDecision(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class MatchSessionStatus(str, Enum):
    PROPOSED = "PROPOSED"
    ACCEPTED_BY_ONE = "ACCEPTED_BY_ONE"
    ACCEPTED = "ACCEPTED"
    VENUE_SELECTED = "VENUE_SELECTED"
    BOOKING_PENDING = "BOOKING_PENDING"
    CONFIRMED = "CONFIRMED"
    CHECKED_IN = "CHECKED_IN"
    RESULT_PENDING = "RESULT_PENDING"
    COMPLETED = "COMPLETED"
    DISPUTED = "DISPUTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class MatchProposalStatus(str, Enum):
    CREATED = "CREATED"
    VENUE_SELECTED = "VENUE_SELECTED"
    BOOKING_PENDING = "BOOKING_PENDING"
    BOOKED = "BOOKED"
    CHECKED_IN = "CHECKED_IN"
    COMPLETED = "COMPLETED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class VenueSearchStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    NO_AVAILABILITY = "NO_AVAILABILITY"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class QueueStatus(str, Enum):
    MATCHED = "MATCHED"
    QUEUED = "QUEUED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class MatchSelectionMode(str, Enum):
    AI = "AI"
    MANUAL = "MANUAL"


class MatchOutcome(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    DRAW = "DRAW"


class AiCapability(str, Enum):
    MATCHMAKING = "MATCHMAKING"
    VENUE_RECOMMENDATION = "VENUE_RECOMMENDATION"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    FEED_RANKING = "FEED_RANKING"
    RAG = "RAG"
    DEMAND_FORECAST = "DEMAND_FORECAST"
    IDENTITY_VERIFICATION = "IDENTITY_VERIFICATION"


class IdentityVerificationStatus(str, Enum):
    PROCESSING = "PROCESSING"
    VERIFIED = "VERIFIED"
    RETRY_REQUIRED = "RETRY_REQUIRED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECTED = "REJECTED"


class IdentityDocumentType(str, Enum):
    CITIZEN_ID_CHIP = "CITIZEN_ID_CHIP"
    IDENTITY_CARD_2024 = "IDENTITY_CARD_2024"
    UNKNOWN = "UNKNOWN"


class AiModelStatus(str, Enum):
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    FAILED = "FAILED"


class EmbeddingSourceType(str, Enum):
    SEMANTIC_DOCUMENT = "SEMANTIC_DOCUMENT"
    KNOWLEDGE_CHUNK = "KNOWLEDGE_CHUNK"


class SearchDocumentType(str, Enum):
    VENUE = "VENUE"
    POST = "POST"
    CLUB_ACTIVITY = "CLUB_ACTIVITY"
    TOURNAMENT = "TOURNAMENT"


class InteractionType(str, Enum):
    VIEW = "VIEW"
    LIKE = "LIKE"
    SAVE = "SAVE"
    COMMENT = "COMMENT"
    SHARE = "SHARE"
    BOOKING = "BOOKING"


class KnowledgeGapStatus(str, Enum):
    OPEN = "OPEN"
    ANSWERED = "ANSWERED"
    DISMISSED = "DISMISSED"


class RecommendationTargetType(str, Enum):
    POST = "POST"
    VENUE = "VENUE"
    EVENT = "EVENT"
    TOURNAMENT = "TOURNAMENT"
