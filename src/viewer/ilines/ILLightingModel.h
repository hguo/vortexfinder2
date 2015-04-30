/* $Id: ILLightingModel.h,v 1.7 2005/10/19 10:52:40 ovidiom Exp $ */

/* forward declarations */
namespace ILines { class ILLightingModel; }

#ifndef _ILLIGHTINGMODEL_H_
#define _ILLIGHTINGMODEL_H_


namespace ILines
{
	/**
	 * @brief Class containing the supported lighting models.
	 */
	class ILLightingModel
	{
	public:
		enum Model
		{
			/** @brief The maximum principle Phong lighting model
			 *         using a \e directional light source. */
			IL_MAXIMUM_PHONG,

			/** @brief The cylinder averaging Phong/Blinn lighting model
			 *         using a \e local light source. */
			IL_CYLINDER_BLINN,

			/** @brief The cylinder averaging Phong lighting model
			 *         using a \e directional light source. */
			IL_CYLINDER_PHONG
		};
	};
}

#endif /* _ILLIGHTINGMODEL_H_ */

