from typing import Annotated

from livekit import agents


class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

    @agents.llm.ai_callable(
        description=(
                "Called when the user asks for any form of visual evaluation or mentions something that could involve "
                "visual analysis, such as an image, video, webcam feed, or even questions about visibility or sight "
                "capabilities"

        )
    )
    async def image(
            self,
            user_msg: Annotated[
                str,
                agents.llm.TypeInfo(
                    description="The user message that triggered this function"
                ),
            ],
    ):
        print(f"###Message triggering vision capabilities: {user_msg}")
        return None

    @agents.llm.ai_callable(
        description=(
                "Called whenever the customer provides any feedback related to the store. "
                "Feedback includes any preferences, suggestions, wishes, complaints, or comments about the store "
                "experience. Examples include, but are not limited to: 'I prefer seeing the price upfront,' 'I don't "
                "like the way this works,' 'It would be better if…,' 'I wish you had…,' etc. "
                "After invoking this function, the agent must respond with an acknowledgment that the feedback has "
                "been received and that it will be forwarded to the appropriate human authorities (e.g., "
                "'human overlords'). The response should reassure the customer that their feedback is valued and will "
                "be acted upon."

        )
    )
    async def store_feedback(
            self,
            user_msg: Annotated[
                str,
                agents.llm.TypeInfo(
                    description="The user message that triggered this function"
                ),
            ],
    ):
        print(f"Storing feedback: {user_msg}")
        return "Noted, we will inform our owners!"

    @agents.llm.ai_callable(
        description=(
                "This method should be called whenever the customer inquires about a specific product. "
                "It retrieves accurate information regarding the availability or location of the product in the store. "
                "The agent must use this method to avoid providing any speculative or incorrect information about the product. "
                "After invoking this method, the agent should inform the customer of the product's status based on the returned result."
        )
    )
    async def check_product(
            self,
            product_name: Annotated[
                str,
                agents.llm.TypeInfo(
                    description="The Name of the product that customer is asking for."
                ),
            ],
    ):
        print(f"Searching for the product: {product_name}")
        return "Product Not available in our offering, inform the customer that human overloards will be informed that user searched for it."
